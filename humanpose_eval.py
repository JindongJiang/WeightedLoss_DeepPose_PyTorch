import argparse
import os
import numpy as np
import csv

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torch.autograd import Variable

import humanpose_net
import humanpose_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch human pose evaluation')
    parser.add_argument('--lsp-root', metavar='DIR',
                        help='path to root of lsp dataset')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--ckpt-dir', default='', type=str, metavar='PATH',
                        help='path to checkpoint')
    parser.add_argument('--num-eval', default=1, type=int, metavar='N',
                        help='How many models to be evaluate')
    parser.add_argument('--no-other-print', action='store_true', default=False,
                        help='disables printing other than pure losses')
    parser.add_argument('--training-eval', action='store_true', default=False,
                        help='Evaluate for training data')
    parser.add_argument('--testing-eval', action='store_true', default=False,
                        help='Evaluate for testing data')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Evaluate all models')
    parser.add_argument('--eval-log', action='store_true', default=False,
                        help='Generate log file')
    parser.add_argument('--pred-log', action='store_true', default=False,
                        help='Write down every output as well as groundtruth')

    args = parser.parse_args()
    args.cuda = (args.cuda and torch.cuda.is_available())
    assert args.training_eval or args.testing_eval, 'Must one or both of --training-eval and --testing-eval'
image_w = 224
image_h = 224

def eval():
    if args.cuda:
        torch.backends.cudnn.benchmarks = True

    model_names = np.array([s for s in os.listdir(args.ckpt_dir) if
                            s.startswith('ckpt_epoch_')])
    model_nums = np.array([float(s.rsplit('_')[2]) for s in model_names])
    sorted_models = model_names[np.argsort(model_nums)]
    sorted_nums = model_nums[np.argsort(model_nums)]
    if args.all:
        args.num_eval = len(sorted_models)

    model = humanpose_net.HumanPoseNet(28, vgg_pretrained=False)
    model.eval()
    if args.cuda:
        model.cuda()

    if args.testing_eval:
        test_data = humanpose_data.LSPDataset(args.lsp_root,
                                              transform=transforms.Compose([humanpose_data.Scale(image_h, image_w),
                                                                            humanpose_data.ToTensor(),
                                                                            humanpose_data.Normalize()]),
                                              phase_train=False,
                                              weighted_loss=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True if args.cuda else False)
    if args.training_eval:
        train_data = humanpose_data.LSPDataset(args.lsp_root,
                                               transform=transforms.Compose([humanpose_data.Scale(image_h, image_w),
                                                                             humanpose_data.ToTensor(),
                                                                             humanpose_data.Normalize()]),
                                               phase_train=True,
                                               weighted_loss=False)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pin_memory=True if args.cuda else False)

    for model_epoch, model_file in zip(sorted_nums[-args.num_eval:], sorted_models[-args.num_eval:]):

        load_ckpt(model, os.path.join(args.ckpt_dir, model_file),
                  args.no_other_print)

        if args.testing_eval:
            evaluation(model, test_loader, model_epoch, args.ckpt_dir, phase_cuda=args.cuda,
                       testing_data=True, phase_eval_log=args.eval_log,
                       phase_pred_log=args.pred_log)
        if args.training_eval:
            evaluation(model, train_loader, model_epoch, args.ckpt_dir, phase_cuda=args.cuda,
                       testing_data=False, phase_eval_log=args.eval_log,
                       phase_pred_log=args.pred_log)


def load_ckpt(model, model_file, no_other_print=False):
    try:
        if not no_other_print:
            print("=> loading checkpoint '{}'".format(model_file))
        if args.cuda:
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if not no_other_print:
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_file, checkpoint['epoch']))
    except FileNotFoundError:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)


def evaluation(model, data_loader, model_epoch, ckpt_dir, phase_cuda, writer=None,
               testing_data=True, phase_eval_log=False, phase_pred_log=False):
    phase_data = 'testing' if testing_data else 'traning'
    total_loss = 0
    num_allpoints = 0
    for batch_idx, sample in enumerate(data_loader):
        image = Variable(sample['image'].cuda() if phase_cuda else sample['image'], volatile=True)
        lm = Variable(sample['landmarks'].cuda() if phase_cuda else sample['landmarks'], volatile=True)
        weight = Variable(sample['weight'].cuda() if phase_cuda else sample['weight'], volatile=True)
        imagefile = sample['imagefile']
        num_allpoints += (weight.clone().cpu().data.numpy() != 0).sum()
        pred = model(image)
        loss = humanpose_net.mse_loss(pred, lm, weight, weighted_loss=False,
                                      size_average=False)
        total_loss += loss.data[0]
        if phase_pred_log:
            with open(os.path.join(ckpt_dir,
                                   'logfile/cp_{:0.2f}_{}_pred.csv'.format(model_epoch, phase_data)),
                      mode='a') as f:
                wtr = csv.writer(f)
                wtr.writerows(np.hstack([np.array(imagefile)[:, np.newaxis],
                                         pred.clone().cpu().data.numpy(),
                                         lm.clone().cpu().data.numpy(),
                                         weight.clone().cpu().data.numpy()]))
    loss = total_loss / num_allpoints
    if phase_eval_log:
        with open(os.path.join(ckpt_dir, 'logfile/{}_eval_log.csv'.format(phase_data)), mode='a') as f:
            f.write('{:0.2f},{}\n'.format(model_epoch, loss))
    if writer is not None:
        writer.add_scalar('Testing data evaluation loss', loss, global_step=int(model_epoch))
    print('epochs {:0.2f} {} loss {:1.5f}'.format(model_epoch, phase_data, loss))


if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.ckpt_dir, 'logfile')):
        os.mkdir(os.path.join(args.ckpt_dir, 'logfile'))
    eval()

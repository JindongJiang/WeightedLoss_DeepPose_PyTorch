import argparse
import os
import time

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import humanpose_net
import humanpose_data
from humanpose_eval import evaluation
from utils import tools

parser = argparse.ArgumentParser(description='PyTorch human pose detection')
parser.add_argument('--lsp-root', metavar='DIR',
                    help='path to root of lsp dataset')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=45, type=int,
                    metavar='N', help='mini-batch size (default: 45)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=30, type=int,
                    metavar='N', help='print batch frequency (default: 30)')
parser.add_argument('--save-freq', '-s', default=300, type=int,
                    metavar='N', help='save batch frequency (default: 300)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=150, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--weighted-loss', '--wl', action='store_true', default=False,
                    help='weighted loss')
parser.add_argument('--weighted-bandwidth', '--wb', default=0.6, type=float,
                    help='bandwidth for loss weight kde')
parser.add_argument('--epochs-per-eval', default=None, type=int,
                    metavar='N', help='Interval of epochs for a testing data evaluation.')
parser.add_argument('--epoch-start-weight', default=None, type=int,
                    metavar='N', help='Epoch on which start weighted loss')

args = parser.parse_args()
args.cuda = (args.cuda and torch.cuda.is_available())
image_w = 224
image_h = 224


def train():
    train_data = humanpose_data.LSPDataset(args.lsp_root,
                                           transform=transforms.Compose([humanpose_data.Scale(image_h, image_w),
                                                                         humanpose_data.RandomHSV((0.8, 1.2),
                                                                                                  (0.8, 1.2),
                                                                                                  (25, 25)),
                                                                         humanpose_data.ToTensor(),
                                                                         humanpose_data.Normalize()]),
                                           phase_train=True,
                                           weighted_loss=args.weighted_loss,
                                           bandwidth=args.weighted_bandwidth)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True if args.cuda else False)
    if args.epochs_per_eval:
        if not os.path.exists(os.path.join(args.ckpt_dir, 'logfile')):
            os.mkdir(os.path.join(args.ckpt_dir, 'logfile'))
        test_data = humanpose_data.LSPDataset(args.lsp_root,
                                              transform=transforms.Compose([humanpose_data.Scale(image_h, image_w),
                                                                            humanpose_data.ToTensor(),
                                                                            humanpose_data.Normalize()]),
                                              phase_train=False,
                                              weighted_loss=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True if args.cuda else False)
    num_alldata = len(train_data)

    model = humanpose_net.HumanPoseNet(train_data.num_keypoints * 2)
    model.train()
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    count = 0
    global_step = 0
    if args.resume:
        count, global_step, args.start_epoch = load_ckpt(model, optimizer, args.resume)

    writer = SummaryWriter(args.summary_dir)

    for epoch in range(int(args.start_epoch), args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr,
                             args.lr_decay_rate, args.lr_epoch_per_decay)
        if args.epochs_per_eval:
            if epoch % args.epochs_per_eval == 0:
                model.eval()
                evaluation(model, test_loader, epoch, args.ckpt_dir, phase_cuda=args.cuda,
                           writer=writer, testing_data=True, phase_eval_log=True)
                model.train()
        for batch_idx, sample in enumerate(train_loader):

            image = Variable(sample['image'].cuda() if args.cuda else sample['image'])
            lm = Variable(sample['landmarks'].cuda() if args.cuda else sample['landmarks'])
            weight = Variable(sample['weight'].cuda() if args.cuda else sample['weight'])
            optimizer.zero_grad()
            pred = model(image)
            loss_log = humanpose_net.mse_loss(pred, lm, weight)
            if args.epoch_start_weight is None or args.epoch_start_weight < epoch:
                loss = humanpose_net.mse_loss(pred, lm, weight, weighted_loss=args.weighted_loss)
            else:
                loss = humanpose_net.mse_loss(pred, lm, weight, weighted_loss=False)
            loss.backward()
            optimizer.step()
            count += image.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0:
                try:
                    time_inter = time.time() - end_time
                    count_inter = count - last_count
                    print_log(global_step, epoch, count, count_inter,
                              num_alldata, loss_log, time_inter)
                    end_time = time.time()
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)
                    grid_image = make_grid(tools.joint_painter(image[:9], pred[:9],
                                                               image_h, image_w), 3)
                    writer.add_image('Predicted image', grid_image, global_step)
                    grid_image = make_grid(tools.joint_painter(image[:9], lm[:9],
                                                               image_h, image_w), 3)
                    writer.add_image('Groundtruth image', grid_image, global_step)
                    writer.add_scalar('MSELoss', loss_log.data[0], global_step=global_step)
                    writer.add_scalar('Weighted MSELoss', loss.data[0], global_step=global_step)
                    last_count = count
                except NameError:
                    end_time = time.time()
                    last_count = count
            if global_step % args.save_freq == 0 or global_step == 1:
                save_ckpt(model, optimizer, global_step, batch_idx, count, args.batch_size,
                          num_alldata, args.weighted_loss, args.weighted_bandwidth)
    save_ckpt(model, optimizer, global_step, batch_idx, count, args.batch_size,
              num_alldata, args.weighted_loss, args.weighted_bandwidth)
    model.eval()
    evaluation(model, test_loader, epoch, args.ckpt_dir, phase_cuda=args.cuda,
               writer=writer, testing_data=True, phase_eval_log=True)

    print("Training completed ")


def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_rate, lr_epoch_per_decay):
    """Sets the learning rate to the initial LR decayed by lr_decay_rate every lr_epoch_per_decay epochs"""
    lr = init_lr * (lr_decay_rate ** (epoch // lr_epoch_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_log(global_step, epoch, count, count_inter, dataset_size, loss, time_inter):
    num_data = count % dataset_size
    num_all = dataset_size
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, num_data, num_all,
                     100. * num_data / num_all, loss.data[0], time_inter, count_inter))


def save_ckpt(model, optimizer, global_step, batch_idx, count,
              batch_size, num_alldata, weighted, bandwidth):
    state = {
        'global_step': global_step,
        'epoch': count / num_alldata,
        'count': count,
        'batch_size': batch_size,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'weighted': weighted,
        'bandwidth': bandwidth,
        'dataset_size': num_alldata,
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(count / num_alldata)
    path = os.path.join(args.ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>35} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if args.cuda:
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.weighted_loss = checkpoint['weighted']
        args.weighted_bandwidth = checkpoint['bandwidth']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        return checkpoint['count'], checkpoint['global_step'], checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    train()

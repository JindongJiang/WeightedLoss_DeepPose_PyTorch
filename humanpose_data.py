import os
import scipy.io
import numpy as np
import skimage.transform
import glob
import torch
from torch.utils.data import Dataset
import scipy.misc
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from torchvision import transforms
from PIL import Image
import csv

lms = None
imagefiles = None
weight = None


class LSPDataset(Dataset):
    def __init__(self, root_dir, transform=None, phase_train=True,
                 weighted_loss=False, bandwidth=50):
        self.scaled_h = 368
        self.scaled_w = 368
        self.map_h = 45
        self.map_w = 45
        self.guassian_sigma = 21
        self.num_keypoints = 14
        self.num_train = 9000
        global lms, imagefiles, weight
        if lms is None or imagefiles is None or weight is None:
            mat_lsp = scipy.io.loadmat(os.path.join(root_dir, 'lsp_dataset/joints.mat'),
                                       squeeze_me=True, struct_as_record=False)['joints']
            mat_lspet = scipy.io.loadmat(os.path.join(root_dir, 'lspet_dataset/joints.mat'),
                                         squeeze_me=True, struct_as_record=False)['joints']
            image_lsp = np.array(glob.glob(os.path.join(root_dir,
                                                        'lsp_dataset/images/*.jpg'), recursive=True))
            image_lspet = np.array(glob.glob(os.path.join(root_dir,
                                                          'lspet_dataset/images/*.jpg'), recursive=True))
            image_nums_lsp = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_lsp])
            image_nums_lspet = np.array([float(s.rsplit('/')[-1][2:-4]) for s in image_lspet])
            sorted_image_lsp = image_lsp[np.argsort(image_nums_lsp)]
            sorted_image_lspet = image_lspet[np.argsort(image_nums_lspet)]

            self.lms = np.append(mat_lspet.transpose([2, 1, 0])[:, :2, :],
                                 # only the x, y coords, not the "block or not" channel
                                 mat_lsp.transpose([2, 0, 1])[:, :2, :],
                                 axis=0)
            self.imagefiles = np.append(sorted_image_lspet, sorted_image_lsp)

            order = []
            with open('./dataorder.txt', 'r') as f:
                rdr = csv.reader(f)
                for row in rdr:
                    order = list(map(int, row))
            self.lms = self.lms[order]
            self.imagefiles = self.imagefiles[order]

            imgs_shape = []
            for img_file in self.imagefiles:
                imgs_shape.append(Image.open(img_file).size)
            imgs_shape = np.array(imgs_shape)[:, :, np.newaxis]
            lms_scaled = self.lms / imgs_shape
            lms_scaled[lms_scaled < 0] = 0
            lms_scaled[lms_scaled >= 1] = 0
            self.weight = (lms_scaled != 0).astype(np.float32)
            self.weight = self.weight[:, 0, :] * self.weight[:, 1, :]
            self.weight = np.append(self.weight, self.weight, axis=1)  # self.num_keypoints * 2
            if weighted_loss and phase_train:
                datas = lms_scaled.copy()[:self.num_train].reshape(self.num_train, -1)
                datas_pca = PCA(n_components=10).fit_transform(datas)
                kde = KernelDensity(bandwidth=bandwidth, kernel='exponential').fit(datas_pca)
                p = np.exp(kde.score_samples(datas_pca))
                p_median = np.median(p)
                p_weighted = p_median / p
                self.weight[:self.num_train] *= p_weighted[:, np.newaxis]
            self.lms = (self.lms - imgs_shape / 2) / (imgs_shape / 2)
            lms = self.lms
            imagefiles = self.imagefiles
            weight = self.weight
        else:
            self.lms = lms
            self.imagefiles = imagefiles
            self.weight = weight

        self.transform = transform
        self.phase_train = phase_train

    def __len__(self):
        if self.phase_train:
            return self.num_train
        else:
            return self.imagefiles.shape[0] - self.num_train

    def __getitem__(self, idx):
        if not self.phase_train:
            idx += self.num_train
        imagefile = str(self.imagefiles[idx])
        image = scipy.misc.imread(imagefile)
        lm = self.lms[idx].flatten()
        weight = self.weight[idx]

        sample = {'image': image, 'landmarks': lm,
                  'weight': weight, 'imagefile': imagefile}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new.astype(np.float32), 'landmarks': sample['landmarks'],
                'weight': sample['weight'], 'imagefile': sample['imagefile']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, weight, imagefile = sample['image'], sample['landmarks'], \
                                              sample['weight'], sample['imagefile']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'weight': torch.from_numpy(weight).float(),
                'imagefile': imagefile}


class Scale(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        image, landmarks, weight, imagefile = sample['image'], sample['landmarks'], \
                                              sample['weight'], sample['imagefile']

        image = skimage.transform.resize(image, (self.height, self.width), preserve_range=True)
        return {'image': image, 'landmarks': landmarks,
                'weight': weight, 'imagefile': imagefile}


class Normalize(object):
    def __call__(self, sample):
        image = sample['image']
        image = image / 255
        normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])(image)
        return {'image': normalized_image, 'landmarks': sample['landmarks'],
                'weight': sample['weight'], 'imagefile': sample['imagefile']}

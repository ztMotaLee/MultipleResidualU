import glob
import torch
import random

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from random import shuffle

class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=[256, 256], random_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')
        
        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        
        if self.mode == 'train':
            self.img_paths_l = sorted(glob.glob(data_source + '/train/low/left/' + '*.*'))
            self.img_paths_r = sorted(glob.glob(data_source + '/train/low/right/' + '*.*'))
            self.gt_paths_l = sorted(glob.glob(data_source + '/train/gt/left/' + '*.*'))
            self.gt_paths_r = sorted(glob.glob(data_source + '/train/gt/right/' + '*.*'))
        else:
            self.img_paths_l = sorted(glob.glob(data_source + '/test/low/left/' + '*.*'))
            self.img_paths_r = sorted(glob.glob(data_source + '/test/low/right/' + '*.*'))
            self.gt_paths_l = sorted(glob.glob(data_source + '/test/gt/left/' + '*.*'))
            self.gt_paths_r = sorted(glob.glob(data_source + '/test/gt/right/' + '*.*'))

    def __getitem__(self, index):
        # read image
        [img_l, img_r, gt_l, gt_r] = [Image.open(x[index % len(x)]) for x in [self.img_paths_l, self.img_paths_r, self.gt_paths_l, self.gt_paths_r]] 
        # merge
        x = np.concatenate((img_l, img_r, gt_l, gt_r), axis=2)

        # augumentation
        if self.mode == 'train':
            # random crop
            h, w = x.shape[0], x.shape[1]
            offset_h = random.randint(0, max(0, h - self.crop[0] - 1))
            offset_w = random.randint(0, max(0, w - self.crop[1] - 1))
            x = x[offset_h:offset_h + self.crop[0], offset_w:offset_w + self.crop[1], :]

            # vertical flip
            if random.random() < 0.5:
                np.flip(x,axis=0)
            # horizontal flip
            if random.random() < 0.5:
                np.flip(x,axis=1)
            # rotate 180
            if random.random() < 0.5:
                np.rot90(x, 2)

        # to tensor
        x = self.to_tensor(x)
        # split
        # print(x.shape)
        img_l, img_r, gt_l, gt_r = x[:3,:,:], x[3:6,:,:], x[6:9,:,:], x[9:,:,:]
        
        return img_l, img_r, gt_l, gt_r

    def __len__(self):
        return max(len(self.img_paths_l), len(self.img_paths_r), len(self.gt_paths_l), len(self.gt_paths_r))
    
    def to_tensor(self, x):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        x = np.transpose(x, (2, 0, 1))/255.0
        x  = torch.from_numpy(x).float()
        return x

class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        if 'real' in data_source:
            self.img_paths_l = sorted(glob.glob(data_source + '/left/' + '*.*'))
            self.img_paths_r = sorted(glob.glob(data_source + '/right/' + '*.*'))
        else:
            self.img_paths_l = sorted(glob.glob(data_source + '/test/low/left/' + '*.*'))
            self.img_paths_r = sorted(glob.glob(data_source + '/test/low/right/' + '*.*'))

    def __getitem__(self, index):
        
        [path_l, path_r] = [x[index % len(x)] for x in [self.img_paths_l, self.img_paths_r]]
        [img_l, img_r] = [Image.open(x) for x in [path_l, path_r]]
        [img_l, img_r] = [self.to_tensor(x) for x in [img_l, img_r]]
        [name_l, name_r] = [x.split("/")[-1] for x in [path_l, path_r]]
        return img_l, img_r, name_l, name_r

    def __len__(self):
        return max(len(self.img_paths_l), len(self.img_paths_r))
    
    
    def to_tensor(self, x):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        x = np.transpose(x, (2, 0, 1))/255.0
        x  = torch.from_numpy(x).float()
        return x

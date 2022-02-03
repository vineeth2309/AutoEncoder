import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

class DatasetClass(Dataset):
    def __init__(self, cfg=None):
        super(DatasetClass, self).__init__()
        self.cfg = cfg
        self.img_files = glob.glob(os.path.join(self.cfg.train_dir, 'image_2', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), (self.cfg.input_width, self.cfg.input_height))
        if self.cfg.augment_data:
            rand_int = np.random.randint(4)
            if rand_int == 0:   # No augmentation
                pass
            elif rand_int == 1: # Horizontal flip
                data = cv2.flip(data, 1)
            elif rand_int == 2: # Vertical flip
                data = cv2.flip(data, 0)
            else:               # Rotation
                angle = np.random.randint(-10, 10)
                w, h = data.shape[1], data.shape[0]
                M =  cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                data = cv2.warpAffine(data, M, (w,h))
        # data = np.expand_dims(data, 0)
        data_tensor = (torch.from_numpy(data).float()/255).permute(2, 0, 1)
        return data_tensor, data_tensor.clone()   # Normalize pixels to lie between [0, 1]

    def __len__(self):
        return len(self.img_files)

class DatasetClass2(Dataset):
    def __init__(self, root='', flag='train', aug=True):
        super(DatasetClass, self).__init__()
        self.flag = flag
        self.aug = aug
        self.img_files = glob.glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4]+'_mask.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.flag == 'test':
            data = np.expand_dims(data, 0)
            data_tensor = torch.from_numpy(data).float()/255
            return data_tensor   # Normalize pixels to lie between [0, 1]
        else:
            mask_path = self.mask_files[index]
            label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if self.flag=='train' and self.aug:
                rand_int = np.random.randint(4)
                if rand_int == 0:   # No augmentation
                    pass
                elif rand_int == 1: # Horizontal flip
                    data, label = cv2.flip(data, 1), cv2.flip(label, 1)
                elif rand_int == 2: # Vertical flip
                    data, label = cv2.flip(data, 0), cv2.flip(label, 0)
                else:               # Rotation
                    angle = np.random.randint(-10, 10)
                    w, h = data.shape[1], data.shape[0]
                    M =  cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    data, label = cv2.warpAffine(data, M, (w,h)), cv2.warpAffine(label, M, (w,h))

            data = np.expand_dims(data, 0)
            data_tensor, label_tensor = torch.from_numpy(data).float()/255, torch.from_numpy(label).long()
            return data_tensor, label_tensor   # Normalize pixels to lie between [0, 1]
            # return data_tensor, data_tensor   # Normalize pixels to lie between [0, 1]

    def __len__(self):
        return len(self.img_files)

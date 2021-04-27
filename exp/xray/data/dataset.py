import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data.imgaug import GetTransforms

np.random.seed(0)

import torchvision
import torchvision.transforms as transforms

import os


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = os.path.join('/home/zhangwd/data/CheXpert-v1.0', fields[0])
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.aug = torchvision.transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            normalize
        ])

    def __len__(self):
        return self._num_image

    def _border_pad(self, image):
        h, w, c = image.shape

        if self.cfg.border_pad == 'zero':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=0.0
            )
        elif self.cfg.border_pad == 'pixel_mean':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=self.cfg.pixel_mean
            )
        else:
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode=self.cfg.border_pad
            )

        return image

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = self.cfg.long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = self.cfg.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)

        image = self._border_pad(image)

        return image

    def __getitem__(self, idx):
        # print(self._image_paths[idx])
        pil_image = Image.open(self._image_paths[idx]).convert('RGB')
        labels = np.array(self._labels[idx]).astype(np.float32)
        image = self.aug(pil_image)

        return (image, labels)

    @staticmethod
    def aug(infile):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        aug = torchvision.transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            normalize
        ])
        pil_image = Image.open(infile).convert('RGB')
        image = aug(pil_image)
        return image


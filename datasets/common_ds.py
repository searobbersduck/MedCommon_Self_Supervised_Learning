import os
import torch
import torchvision

from glob import glob
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torchio as tio


class CommonDS(Dataset):
    def __init__(self, image_files, image_shape=[128,128,128], transforms=None):
        self.image_files = image_files

        if transforms:
            self.transforms = transforms
        else:
            default_transform = tio.Compose([
                tio.RandomFlip(axes=[0,1,2]), 
                tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
                tio.CropOrPad((image_shape[0], image_shape[1], image_shape[2])),            # tight crop around brain
                tio.RandomBlur(p=0.25),                    # blur 25% of times
                tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
                tio.OneOf({                                # either
                    tio.RandomAffine(): 0.8,               # random affine
                    tio.RandomElasticDeformation(): 0.2,   # or random elastic deformation
                }, p=0.8),                                 # applied to 80% of images
                tio.RandomBiasField(p=0.3),                # magnetic field inhomogeneity 30% of times
                tio.OneOf({                                # either
                    tio.RandomMotion(): 1,                 # random motion artifact
                    tio.RandomSpike(): 2,                  # or spikes
                    tio.RandomGhosting(): 2,               # or ghosts
                }, p=0.5), 
            ])
            self.transforms = default_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_file = self.image_files[item]
        image = sitk.ReadImage(image_file)
        arr = sitk.GetArrayFromImage(image)

        arr = np.expand_dims(arr, axis=0)

        image_one = self.transforms(arr)
        image_two = self.transforms(arr)

        image_one = torch.from_numpy(image_one).float()
        image_two = torch.from_numpy(image_two).float()

        return image_one, image_two, image_file

class CardiacDS(CommonDS):
    def __init__(self, root, image_shape=[128,128,128], transforms=None):
        pids = os.listdir(root)
        self.image_files = [os.path.join(root, pid, 'cropped_cta.nii.gz') for pid in pids]
        super().__init__(self.image_files, image_shape, transforms)


def test_CardiacDS():
    root = '/fileser/zhangwd/data/cardiac/cta2mbf/20201216/5.mbf_myocardium'
    ds = CardiacDS(root, [64, 64, 64])
    dataloader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=False, shuffle=True, drop_last=True)

    for index, (image_one, image_two, _) in enumerate(dataloader):
        print(image_one.shape)


if __name__ == '__main__':
    test_CardiacDS()
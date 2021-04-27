import torch
import torch.nn as nn
import torchvision


import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
print(root)
sys.path.append('../../')
sys.path.append(root)
sys.path.append(os.path.join(root, 'external_lib/Chexpert/'))
sys.path.append(os.path.join(root, 'external_lib/pytorch-grad-cam'))

import pytorch_grad_cam
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import DRModel

import cv2
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from einops import rearrange, reduce, repeat

from external_lib.Chexpert.data.dataset import ImageDataset

import torch.nn as nn

class XrayModel(nn.Module):
    def __init__(self, num_classes=5, pretrained_file=None):
        super().__init__()
        self.pretrained_file = pretrained_file
        self.num_classes = num_classes
        self.dr_model = DRModel(num_classes)
        if not pretrained_file:
            state_dict = torch.load(pretrained_file, map_location='cpu')
            self.dr_model.load_state_dict(state_dict, strict=False)

        target_layer = self.dr_model.features[7][-1]
        self.cam = GradCAM(model=self.dr_model, target_layer=target_layer, use_cuda=True)

    def forward(self, input):
        output = self.dr_model(input)
        grayscale_cams = []
        for i in range(self.num_classes):
            grayscale_cam = self.cam(input_tensor=input, target_category=i)
            grayscale_cams.append(grayscale_cam)
        grayscale_cams = torch.as_tensor(grayscale_cams)
        return output, grayscale_cams
        
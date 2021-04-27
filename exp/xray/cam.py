import torch
import torch.nn as nn
import torchvision


import os
import sys
sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../../external_lib/Chexpert/')
sys.path.append('../../../external_lib/pytorch-grad-cam')

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

pretrained_file = './checkpoints/best.pth'
dr_model = DRModel(5)
state_dict = torch.load(pretrained_file, map_location='cpu')
dr_model.load_state_dict(state_dict, strict=False)

target_layer = dr_model.features[7][-1]

image_file = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/train/patient00005/study1/view1_frontal.jpg'
cv_img = cv2.imread(image_file)

print(cv_img.shape)
cv_img = cv2.resize(cv_img, (512,512))
print(cv_img.shape)
plt.imshow(cv_img)

np_img = rearrange(cv_img, 'h w c -> c h w')
print(np_img.shape)

input_tensor = torch.from_numpy(np_img).unsqueeze(0)
# input_tensor = torch.from_numpy(np_img)
input_tensor = input_tensor.float()
print(input_tensor.dtype)
print(input_tensor.shape)

cam = GradCAM(model=dr_model, target_layer=target_layer, use_cuda=True)

grayscale_cam = cam(input_tensor=input_tensor, target_category=1)

visualization = show_cam_on_image(cv_img, grayscale_cam)

cv2.imwrite('./tmp.png', visualization)

print('hello world!')

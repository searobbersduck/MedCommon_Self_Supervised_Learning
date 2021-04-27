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

from jit_model import XrayModel


# export model: 5-classifier

pretrained_file = '../checkpoints/epoch_0_0.8103.pth'
num_classes=5
model = DRModel(num_classes)
model.load_state_dict(torch.load(pretrained_file, map_location='cpu'))
dummy_input = torch.rand(1, 3, 512, 512)
trace_model = torch.jit.trace(model, dummy_input)
trace_model.save('xray_classifier.pt')

# config.pbtxt
'''

name: "xray_classifier"
platform: "pytorch_libtorch"
max_batch_size: 256
input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 3,-1,-1 ]
  }
]
output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [5]
  }
]
instance_group [
  {
    kind: KIND_GPU
  }
]

optimization { execution_accelerators {
  gpu_execution_accelerator : [
    { name : "auto_mixed_precision" }
  ]
}}

'''



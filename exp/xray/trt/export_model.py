import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

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

from data.dataset import ImageDataset

import numpy as np
from tqdm import tqdm

from sklearn import metrics

def train(data_loader, model, criterion, num_classes, display=10, mode='train', epoch=0):
    total_prob = [np.array([]) for i in range(num_classes)]
    total_gt = [np.array([]) for i in range(num_classes)]
    total_pred = [np.array([]) for i in range(num_classes)]
    files = []
    for index, (images, labels, names) in tqdm(enumerate(data_loader)):
        files += list(names)
        if mode == 'train':
            model.train()
        else:
            model.eval()
        output = model(images.cuda())
        
        loss = 0
        for task_id in range(num_classes):
            loss_t = criterion(output[:,task_id], labels[:, task_id].cuda())
            loss += loss_t
            total_prob[task_id] = np.append(total_prob[task_id], output[:,task_id].detach().cpu().numpy())
            total_gt[task_id] = np.append(total_gt[task_id], labels[:, task_id])
            total_pred[task_id] = np.append(total_pred[task_id], output[:, task_id].view(-1).ge(0.5).float().cpu().numpy())
        # pred_label = torch.sigmoid(output.view(-1)).ge(0.5).float().cpu()
        pred_label = output.view(-1).ge(0.5).float().cpu()
        target = labels.view(-1)
        
        acc = (target == pred_label).float().sum() / len(pred_label)


        if index % display == 0:
            print('Epoch:[{}][{}/{}]\t'.format(epoch, index, len(data_loader)),'loss:{:.4f}'.format(loss.detach().cpu().item()), '\t', 'acc:{:.3f}'.format(acc))
            # print(total_prob)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    tmp_pred = np.array(total_pred).flatten()
    tmp_gt = np.array(total_gt).flatten()
    total_acc = (tmp_gt == tmp_pred).sum() / len(tmp_gt)
    
    auclist = []
    for i in range(num_classes):
        y_pred = total_prob[i]
        y_true = total_gt[i]
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auclist.append(auc)

    # print(total_prob[0].round(3))
    return total_prob, total_pred, total_gt, total_acc, auclist, files
   

infile = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/valid/patient64609/study1/view1_frontal.jpg'
image = ImageDataset.aug(infile)
image = image.unsqueeze(0)

# export model: 5-classifier

pretrained_file = '../checkpoints/epoch_0_0.8077.pth'

from easydict import EasyDict as edict
import json
from torch.utils.data import DataLoader

class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
print(root)
cfg_path = os.path.join(root, 'external_lib/Chexpert/config/lse_fpa.json')
with open(cfg_path) as f:
    cfg = edict(json.load(f))
val_label_path = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/valid.csv'
val_ds = ImageDataset(val_label_path, cfg, mode='val')
val_data_loader = DataLoader(val_ds, batch_size=4, num_workers=8, shuffle=True, pin_memory=False)

num_classes=5
model = DRModel(num_classes)
model.load_state_dict(torch.load(pretrained_file, map_location='cpu'))

# criterion = torch.nn.BCELoss()
# total_prob, total_pred, total_gt, acc,auc_list,files = train(val_data_loader, model.cuda(), criterion, num_classes, 10, 'val')

# print('best acc:\t{:.4f}'.format(acc))
# for i in range(num_classes):
#   print('{} auc:\t{:.4f}'.format(class_names[i], auc_list[i]))

# pid = infile.split('/')[-3]
# pid_index = [i.split('/')[-3] for i in files].index(pid)
# indexed_probs = np.array(total_prob)[:,pid_index]

# print(indexed_probs)

model.cuda()
model.eval()
probs = model(image.cuda())
print(probs)

# print(probs.detach().cpu().numpy() == indexed_probs)

model.cpu()
dummy_input = torch.rand(1, 3, 512, 512)
trace_model = torch.jit.trace(model, dummy_input)
trace_model.save('xray_classifier.pt')


# load model
loaded_model = torch.jit.load('xray_classifier.pt')
probs_pt = loaded_model(image)
print(probs_pt)

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



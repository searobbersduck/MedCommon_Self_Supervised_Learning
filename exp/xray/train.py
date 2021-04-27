import os
import sys

from easydict import EasyDict as edict
import json

import torch
from torch.utils.data import DataLoader

from data.dataset import ImageDataset

import torch
import torch.nn as nn
import torchvision

from model import DRModel

from tqdm import tqdm
import numpy as np
from sklearn import metrics

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
print(root)
cfg_path = os.path.join(root, 'external_lib/Chexpert/config/lse_fpa.json')
label_path = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/train.csv'
ckpt_path = './moco/checkpoints/resnet18_xray.pth.tar'
# cls_ckpt_path = './checkpoints/best.pth'
cls_ckpt_path = './checkpoints/epoch_0_0.8103.pth'
# cls_ckpt_path = None
n_epochs = 100
num_classes = 5




with open(cfg_path) as f:
    cfg = edict(json.load(f))

ds = ImageDataset(label_path, cfg, mode='train')
data_loader = DataLoader(ds, batch_size=256, num_workers=8, shuffle=True, pin_memory=True)

val_label_path = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/valid.csv'
val_ds = ImageDataset(val_label_path, cfg, mode='val')
val_data_loader = DataLoader(val_ds, batch_size=4, num_workers=8, shuffle=True, pin_memory=True)


pretrained = ckpt_path
model = DRModel(num_classes, pretrained)
if cls_ckpt_path:
    model.load_state_dict(torch.load(cls_ckpt_path, map_location='cpu'))
model.cuda()
model = torch.nn.DataParallel(model)

# criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']

def train(data_loader, model, criterion, num_classes, display=10, mode='train'):
    total_prob = [np.array([]) for i in range(num_classes)]
    total_gt = [np.array([]) for i in range(num_classes)]
    total_pred = [np.array([]) for i in range(num_classes)]
    for index, (images, labels) in tqdm(enumerate(data_loader)):
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
    return total_prob, total_pred, total_gt, total_acc, auclist
    

best_acc = 0

for epoch in range(n_epochs):
    train(data_loader, model, criterion, num_classes, 10, 'train')
    _,_,_,acc,auc_list = train(val_data_loader, model, criterion, num_classes, 10, 'val')
    if best_acc < acc:
        print('best acc:\t{:.4f}'.format(acc))
        for i in range(num_classes):
            print('{} auc:\t{:.4f}'.format(class_names[i], auc_list[i]))
        outdir = './checkpoints'
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(outdir, 'epoch_{}_{:.4f}.pth'.format(epoch, acc)))


print('hello world!')

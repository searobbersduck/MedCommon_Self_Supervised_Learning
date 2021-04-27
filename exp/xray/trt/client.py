import numpy as np
import sys
import gevent.ssl

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import SimpleITK as sitk
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from data.dataset import ImageDataset
from sklearn import metrics


# 1. 尝试连接服务
url = '10.100.37.100:8700'
verbose = False
model_name = 'xray_classifier'

triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose)

# img = np.random.rand(1,3,512,512)
# img = np.array(img, dtype=np.float32)

infile = '/home/zhangwd/data/CheXpert-v1.0/CheXpert-v1.0-small/valid/patient64609/study1/view1_frontal.jpg'

class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']

def inference(infile):
    image = ImageDataset.aug(infile)
    image = image.unsqueeze(0)
    image = image.numpy()
    # image = np.expand_dims(image, 0)
    print(image.shape)

    inputs = []
    inputs.append(httpclient.InferInput('INPUT__0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True))

    results = triton_client.infer(model_name,
                                    inputs,
                                    outputs=outputs)
    probs = results.as_numpy('OUTPUT__0')[0]

    print(probs)

# inference(infile)

def inference_batch(image):
    inputs = []
    inputs.append(httpclient.InferInput('INPUT__0', image.shape, "FP32"))
    inputs[0].set_data_from_numpy(image, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True))

    results = triton_client.infer(model_name,
                                    inputs,
                                    outputs=outputs)
    probs = results.as_numpy('OUTPUT__0')
    return probs


# batch inference
from easydict import EasyDict as edict
import json
from torch.utils.data import DataLoader

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
print(root)
cfg_path = os.path.join(root, 'external_lib/Chexpert/config/lse_fpa.json')
with open(cfg_path) as f:
    cfg = edict(json.load(f))
val_label_path = '/data/medical/external/xray/CheXpert/CheXpert-v1.0-small/valid.csv'
val_ds = ImageDataset(val_label_path, cfg, mode='val')
val_data_loader = DataLoader(val_ds, batch_size=128, num_workers=8, shuffle=True, pin_memory=False)

num_classes = 5
total_probs = [np.array([]) for i in range(num_classes)]
total_gt = [np.array([]) for i in range(num_classes)]
total_pred = [np.array([]) for i in range(num_classes)]
files = []
for index, (images, labels, names) in enumerate(val_data_loader):
    files += list(names)
    probs = inference_batch(images.numpy())
    for i in range(num_classes):
        total_probs[i] = np.append(total_probs[i], probs[:,i])
        total_pred[i] = np.append(total_pred[i], np.array(np.greater_equal(probs[:,i], 0.5), dtype=np.float32))
        total_gt[i] = np.append(total_gt[i], labels.numpy()[:,i])

auclist = []
acclist = []
for i in range(num_classes):
    acc = (total_gt[i] == total_pred[i]).sum() / len(total_gt[i])
    acclist.append(acc)

    y_pred = total_probs[i]
    y_true = total_gt[i]
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    auclist.append(auc)

for i in range(num_classes):
    print('{} auc:\t{:.4f}\t\tacc:\t{:.4f}'.format(class_names[i], auclist[i], acclist[i]))
    for j in range(len(total_gt[i])):
        if total_gt[i][j] == 1 and total_gt[i][j] == total_pred[i][j]:
            print(files[i])
    print('\n')   

print('\n====> result:')
for i in range(len(total_gt[0])):
    print('{}\t{}\t{}'.format(files[i], total_probs[i], total_gt[i]))    
        

print('hello world')

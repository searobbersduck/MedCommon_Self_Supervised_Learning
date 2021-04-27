# 利用自监督训练X光图片分类

## 运行环境
```
conda create --name moco_xray --file requirements.txt
conda activate moco_xray
```

## 数据集

[ChexXpert: A Large Chest X-Ray Dataset And Competition](https://stanfordmlgroup.github.io/competitions/chexpert/)

这是一个开源的xray图片的数据集，包含了200w+的图片

数据下载后的存储路径, 这里为了便于试验采用了小数据集（即xray图片的size为512x512）：

```
    ~/data/CheXpert-v1.0/CheXpert-v1.0-small$ tree -L 1
    .
    ├── train
    ├── train.csv
    ├── valid
    └── valid.csv

```

本例中设计到xray相关诊断疾病：

```
Cardiomegaly auc:	0.8143
Edema auc:	0.9223
Consolidation auc:	0.9145
Atelectasis auc:	0.7709
Pleural_Effusion auc:	0.9192
```

## 利用自监督预训练模型参数

具体请参见:[自监督预训练模型参数]('./moco/readme.md')

## 微调模型

`python train.py`

基本5个epoch就能得到最好的模型，这里选用`epoch 0`的结果进行下一步骤，模型存放在`./checkpoints`
```
Epoch:[0][0/59]	 loss:3.4938 	 acc:0.650
Epoch:[0][10/59]	 loss:1.4220 	 acc:0.900
Epoch:[0][20/59]	 loss:1.7769 	 acc:0.750
Epoch:[0][30/59]	 loss:1.5458 	 acc:0.850
Epoch:[0][40/59]	 loss:1.9232 	 acc:0.850
Epoch:[0][50/59]	 loss:2.1552 	 acc:0.750
best acc:	0.8077
Cardiomegaly auc:	0.8143
Edema auc:	0.9223
Consolidation auc:	0.9145
Atelectasis auc:	0.7709
Pleural_Effusion auc:	0.9192

Epoch:[2][0/59]	 loss:2.3978 	 acc:0.850
Epoch:[2][10/59]	 loss:0.7635 	 acc:0.950
Epoch:[2][20/59]	 loss:2.7504 	 acc:0.800
Epoch:[2][30/59]	 loss:1.7090 	 acc:0.800
Epoch:[2][40/59]	 loss:2.4728 	 acc:0.750
Epoch:[2][50/59]	 loss:0.9292 	 acc:0.900
best acc:	0.8179
Cardiomegaly auc:	0.8203
Edema auc:	0.9297
Consolidation auc:	0.8247
Atelectasis auc:	0.8003
Pleural_Effusion auc:	0.9308


Epoch:[3][0/59]	 loss:2.6292 	 acc:0.900
Epoch:[3][10/59]	 loss:2.3264 	 acc:0.800
Epoch:[3][20/59]	 loss:2.0473 	 acc:0.850
Epoch:[3][30/59]	 loss:1.4072 	 acc:0.850
Epoch:[3][40/59]	 loss:1.4376 	 acc:0.900
Epoch:[3][50/59]	 loss:3.8339 	 acc:0.600
best acc:	0.8068
Cardiomegaly auc:	0.7457
Edema auc:	0.9115
Consolidation auc:	0.8191
Atelectasis auc:	0.8616
Pleural_Effusion auc:	0.9043
```

## 调用模型

`./notebook/xray_cam.ipynb`

## 导出模型

将模型导出`trt`模型，利用`triton inference server`进行挂载

导出模型：`python ./trt/export_model.py`, 保存模型：`./trt/xray_classifier.pt`

将模型放置于triton server 模型仓库: `cp ./trt/xray_classifier.pt $tritonserver/models/xray_classifier/1/model.pt`

## triton inference server挂载模型

## 通过triton inference server挂载的模型进行推断

推断模型：`python ./trt/client.py`






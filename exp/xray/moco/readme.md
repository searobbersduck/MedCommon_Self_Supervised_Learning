# 自监督训练xray预训练模型

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

## 训练

`./train.sh`

实例中，用到的架构为`resnet18`，这里可以根据需要调整；如果不修改模型代码，默认支持的模型为`pytorch`的`torchvision`中支持的模型；

如下为训练过程：
```
Epoch: [218][ 860/1745] Time  0.379 ( 0.415)    Data  0.000 ( 0.025)    Loss 4.5344e-01 (4.5096e-01)    Acc@1 100.00 ( 95.36)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 870/1745] Time  0.380 ( 0.415)    Data  0.000 ( 0.025)    Loss 3.7083e-01 (4.5080e-01)    Acc@1 100.00 ( 95.40)   Acc@5 100.00 ( 99.79)
Epoch: [218][ 880/1745] Time  0.423 ( 0.415)    Data  0.000 ( 0.025)    Loss 3.6538e-01 (4.4994e-01)    Acc@1 100.00 ( 95.45)   Acc@5 100.00 ( 99.79)
Epoch: [218][ 890/1745] Time  0.395 ( 0.415)    Data  0.000 ( 0.025)    Loss 3.7501e-01 (4.4946e-01)    Acc@1 100.00 ( 95.50)   Acc@5 100.00 ( 99.79)
Epoch: [218][ 900/1745] Time  0.390 ( 0.415)    Data  0.000 ( 0.025)    Loss 4.0426e-01 (4.4820e-01)    Acc@1  96.88 ( 95.55)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 910/1745] Time  0.392 ( 0.415)    Data  0.000 ( 0.025)    Loss 3.1411e-01 (4.4779e-01)    Acc@1 100.00 ( 95.59)   Acc@5 100.00 ( 99.79)
Epoch: [218][ 920/1745] Time  0.434 ( 0.414)    Data  0.000 ( 0.024)    Loss 3.0633e-01 (4.4695e-01)    Acc@1 100.00 ( 95.63)   Acc@5 100.00 ( 99.79)
Epoch: [218][ 930/1745] Time  0.393 ( 0.414)    Data  0.014 ( 0.024)    Loss 2.5896e-01 (4.4609e-01)    Acc@1 100.00 ( 95.67)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 940/1745] Time  0.560 ( 0.415)    Data  0.000 ( 0.024)    Loss 3.4697e-01 (4.4503e-01)    Acc@1 100.00 ( 95.72)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 950/1745] Time  0.576 ( 0.416)    Data  0.000 ( 0.024)    Loss 4.5436e-01 (4.4451e-01)    Acc@1  96.88 ( 95.75)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 960/1745] Time  0.558 ( 0.418)    Data  0.000 ( 0.024)    Loss 4.1997e-01 (4.4428e-01)    Acc@1 100.00 ( 95.79)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 970/1745] Time  0.564 ( 0.419)    Data  0.000 ( 0.024)    Loss 3.2954e-01 (4.4408e-01)    Acc@1 100.00 ( 95.82)   Acc@5 100.00 ( 99.80)
Epoch: [218][ 980/1745] Time  0.543 ( 0.420)    Data  0.000 ( 0.024)    Loss 3.6647e-01 (4.4347e-01)    Acc@1 100.00 ( 95.86)   Acc@5 100.00 ( 99.81)
Epoch: [218][ 990/1745] Time  0.464 ( 0.422)    Data  0.000 ( 0.024)    Loss 2.8755e-01 (4.4228e-01)    Acc@1 100.00 ( 95.90)   Acc@5 100.00 ( 99.81)
Epoch: [218][1000/1745] Time  0.378 ( 0.421)    Data  0.000 ( 0.024)    Loss 4.3126e-01 (4.4193e-01)    Acc@1 100.00 ( 95.94)   Acc@5 100.00 ( 99.81)

```

## 保存预训练参数

通过自监督预训练的结果保存在`./checkpoints/resnet18_xray.pth.tar`

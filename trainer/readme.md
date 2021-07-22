
<br><br>
---
## 训练MBF模型

```
CUDA_VISIBLE_DEVICES=2,3,4,5 python main_moco.py -a resnet10 --lr 0.03 --batch-size 4 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 192  --epochs 50000 /data/medical/cardiac/cta2mbf/ssl/cropped_ori --ds MBF --p 10 --crop_size 352 352 160 --resume ./MBF/checkpoint_47100.pth.tar
```

数据为配准后，并且crop出心脏部分的mbf数据，数据格式：
```
cropped_ori$ tree -L 1
.
├── 1023293_cropped_mbf.nii.gz
├── 1037361_cropped_mbf.nii.gz
├── 1051819_cropped_mbf.nii.gz
├── 1063502_cropped_mbf.nii.gz
├── 1069558_cropped_mbf.nii.gz
├── 1075085_cropped_mbf.nii.gz

```

模型架构`resnet10`,预训练好的模型：

```
trainer/MBF/mbf_ssl.pth.tar
```


<br><br>
---
## 训练冠脉或心脏模型

```
CUDA_VISIBLE_DEVICES=2,3,4,5 python main_moco.py -a resnet18 --lr 0.03 --batch-size 4 --dist-url 'tcp://localhost:10004' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 512  --epochs 10000 /data/medical/cardiac/seg/coronary/coronary_ori_256/images --ds DetectionCoronary --resume /home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/DetectionCoronary/checkpoint_9990.pth.tar
```

数据为冠脉心脏数据，resize到256大小，数据格式：
```
/data/medical/cardiac/seg/coronary$ tree -L 1
.
├── coronary_cropped_by_mask
├── coronary_ori
├── coronary_ori_256
```

```
/data/medical/cardiac/seg/coronary/coronary_ori_256/images$ tree -L 1
.
├── 1.2.392.200036.9116.2.2054276706.1536112085.7.1121200003.1.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1558914765.13.1071700006.1.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1568850311.7.1124000003.1.nii.gz
├── 1.2.392.200036.9116.2.2054276706.1578352061.11.1177300004.1.nii.gz

```

模型架构`resnet18`,预训练好的模型，这里模型太大暂不上传了：

```
trainer/DetectionCoronary/checkpoint_9990.pth.tar
```

<br><br>
----

## 训练XRay模型

```
CUDA_VISIBLE_DEVICES=6,7 python main_moco.py -a resnet10 --lr 0.03 --batch-size 2 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 512  --epochs 10000 /data/medical/external/xray/CheXpert/CheXpert-v1.0
```

模型路径：
```
exp/xray/moco/checkpoints/resnet18_xray.pth.tar
```



# 使用示例

1. [load 实例](https://github.com/searobbersduck/MedCommon/blob/main/utils/ssl_utils.py)
2. [利用训练好的自监督模型做特征提取](https://github.com/searobbersduck/MedCommon/blob/main/gan/models/pix2pix_3d_model.py)
    ```
    if opt.ssl_sr:
        if self.opt.ssl_arch is not None and self.opt.ssl_pretrained_file is not None:
            self.features_extractor = SSL_Utils.load_ssl_model(self.opt.ssl_arch, self.opt.ssl_pretrained_file)
            self.features_extractor = torch.nn.Sequential(*list(self.features_extractor.children())[:1])
            self.features_extractor.to(self.device)
            self.loss_names.append('SR')
    ```
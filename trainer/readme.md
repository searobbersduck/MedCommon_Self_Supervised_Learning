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

预训练好的模型：

```

```
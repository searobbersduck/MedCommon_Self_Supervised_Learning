## 训练MBF模型

```
CUDA_VISIBLE_DEVICES=2,3,4,5 python main_moco.py -a resnet10 --lr 0.03 --batch-size 4 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 192  --epochs 50000 /data/medical/cardiac/cta2mbf/ssl/cropped_ori --ds MBF --p 10 --crop_size 352 352 160 --resume ./MBF/checkpoint_47100.pth.tar
```

数据格式：
```

```
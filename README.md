# MedCommon_Self_Supervised_Learning
a common self supervised learning solution for medical image

## external lib

```
# backbone module
git submodule add https://github.com/kenshohara/3D-ResNets-PyTorch.git

# byol pytorch
git submodule add https://github.com/lucidrains/byol-pytorch.git

# moco 
git submodule add https://github.com/facebookresearch/moco.git

# torchio
git submodule add https://github.com/searobbersduck/torchio.git

```

## Preparation

```
conda create --name pytorch1.8 --file requirements.txt
conda activate pytorch1.8
```

## Demo

[利用自监督训练X光图片分类并高亮关注区域](./exp/xray/readme.md)

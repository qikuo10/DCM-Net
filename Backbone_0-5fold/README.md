# 特征文件目录

此目录用于存放预提取的骨干网络特征文件。

## 目录结构

```
Backbone_0-5fold/
└── data_feats/
    └── swinB384_features/
        ├── k0_train_feats.pkl   # 第0折训练集特征
        ├── k0_feats.pkl         # 第0折测试集特征
        ├── k1_train_feats.pkl
        ├── k1_feats.pkl
        └── ...
```

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

## 特征格式

特征以 pickle 文件存储，包含字典 `{视频ID: numpy数组}`，数组形状为 `(帧数, 特征维度)`。

- Swin-Tiny: 768 维
- Swin-Base: 1024 维
- Swin-Base 384: 1024 维

特征提取由 CurConMix 完成，请参考其文档获取提取方法。

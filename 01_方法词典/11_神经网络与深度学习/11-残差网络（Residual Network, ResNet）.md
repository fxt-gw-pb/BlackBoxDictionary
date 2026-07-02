---
title: 残差网络
english_name: Residual Network, ResNet
slug: residual-network-resnet
aliases: [ResNet, residual network, "残差网络（Residual Network, ResNet）"]
category: 神经网络与深度学习
subcategory: 深层卷积网络
tags: [医学统计, 数据科学, 深度学习, 图像分析]
status: 已建
difficulty: advanced
question_type: 深层图像特征学习
data_type: [图像数据, 三维影像数据, 栅格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [torch, torchvision]
r_packages: [keras]
---

# 残差网络（Residual Network, ResNet）

## 1. 方法概览

### 1.1 定义

ResNet 是在深层 CNN 中引入残差连接的网络架构。它通过快捷连接把输入直接加到后续层输出上，从而缓解深层网络训练中的退化与梯度传播困难。

### 1.2 它主要解决什么问题

- 研究问题：如何在网络很深时仍保持可训练性和较好的泛化。
- 适用任务：图像分类、检测、分割、迁移学习、医学影像表征提取。
- 常见医学场景：CT/MRI 分类、病理图像分型、眼底和皮肤图像识别。

### 1.3 直觉理解

ResNet 让某些层学的不是完整映射，而是“相对输入还需要补充什么”。如果额外变化不多，网络可以更容易学到接近恒等映射的结果，因此深层训练更稳定。

## 2. 数学形式

### 2.1 核心公式

残差块可写为：

$$
y = F(x, W) + x
$$

若输入输出维度不一致，则常用投影捷径：

$$
y = F(x, W) + W_s x
$$

其中 $F(x, W)$ 表示卷积、归一化和激活构成的残差分支。

### 2.2 参数或统计量含义

- 残差分支 $F$：学习相对输入的增量表示。
- 捷径连接：直接传递输入，改善梯度流动。
- block 深度与通道数：决定网络容量和表征能力。

### 2.3 关键假设

- 深层卷积特征对任务有价值。
- 训练数据足够支持较深网络。
- 残差连接能改善优化而非仅增加参数。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：二维或三维图像张量。
- 因变量类型：分类、回归、分割等视觉任务标签。
- 数据结构：图像张量。
- 是否适合高维数据：适合高维图像输入。
- 是否适合缺失较多数据：需先处理图像缺损或缺失切片。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：若是图像序列，可与时序结构结合。

### 3.2 示例表格

以眼底图像分级为例：

| ImageID | ImagePath | Eye | DRGrade |
| --- | --- | --- | --- |
| F001 | fundus_001.png | left | 2 |
| F002 | fundus_002.png | right | 0 |
| F003 | fundus_003.png | left | 3 |
| F004 | fundus_004.png | right | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：图像张量和视觉任务标签。
- 关键变量：网络深度、残差块类型、学习率、数据增强方案。
- 需要预处理的内容：归一化、裁剪、增强、病人级划分。

#### 产出

- 模型对象/统计结果：训练好的残差网络和验证指标。
- 参数估计：各残差块与分类头权重。
- 预测结果：类别概率、回归值或中间特征嵌入。
- 不确定性指标：外部验证 AUC、灵敏度、校准和错误案例复核。

## 4. 适用场景

- 适合：需要较深视觉特征抽取或迁移学习的影像任务。
- 不适合：样本极少且无法借助预训练时。
- 使用前需要特别检查的点：病人级划分、中心偏差、标签噪声、是否过度依赖预训练域。

## 5. 实现

### 5.1 Python

常用包：

- `torch`
- `torchvision`

```python
from torchvision.models import resnet18

model = resnet18(weights=None, num_classes=2)
```

### 5.2 R

常用包：

- `keras`

```r
# R 中可通过 keras 的 application_resnet* 预训练模型做迁移学习。
```

## 6. 结果如何解释

- 核心结果看什么：外部图像集表现、不同病灶等级下召回率、错误样本模式。
- 每个主要参数如何解释：深度越大容量越高，但也更依赖数据和正则化。
- 临床或医学意义如何表达：更适合作为高性能视觉识别骨架，需配合热图和错误案例审核。
- 常见误读：ResNet 深并不自动代表更好，数据质量和验证设计同样关键。

## 7. 推荐可视化

- 训练/验证损失曲线。
- 混淆矩阵或 ROC 曲线。
- Grad-CAM 热图。

## 8. 优势、局限与常见坑

### 优势

- 深层训练更稳定。
- 迁移学习生态成熟。
- 在医学影像任务中常是强基线。

### 局限

- 仍需要较多算力和较好标注。
- 可解释性有限。
- 对域偏移较敏感。

### 常见坑

- 用图像级随机划分代替病人级划分。
- 预训练后不检查目标域差异。
- 只报告平均准确率，不看关键亚组。

## 9. 与相近方法的区别

- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：ResNet 是深层 CNN 的具体残差化架构。
- 和 [[AlexNet（AlexNet）]] 的区别：AlexNet 更早、更浅，ResNet 重点解决深层训练退化。
- 和 [[DenseNet（Densely Connected Convolutional Network, DenseNet）]] 的区别：DenseNet 采用密集连接，复用所有前层特征。

## 10. 医学研究中的典型应用

- 眼底、胸片、病理切片分类。
- 影像表征提取和迁移学习。
- 作为分割和检测网络骨干。

## 11. 相关方法

- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[AlexNet（AlexNet）]]
- [[DenseNet（Densely Connected Convolutional Network, DenseNet）]]
- [[Transformer（Transformer）]]

## 12. 参考资料

- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770-778.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- torchvision Developers. ResNet models. [https://pytorch.org/vision/stable/models/resnet.html](https://pytorch.org/vision/stable/models/resnet.html) （访问日期：2026-07-02）

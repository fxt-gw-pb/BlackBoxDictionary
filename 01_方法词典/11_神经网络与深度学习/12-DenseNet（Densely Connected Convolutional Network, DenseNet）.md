---
title: DenseNet
english_name: Densely Connected Convolutional Network, DenseNet
slug: densely-connected-convolutional-network-densenet
aliases: [DenseNet, densely connected convolutional network, "DenseNet（Densely Connected Convolutional Network, DenseNet）"]
category: 神经网络与深度学习
subcategory: 深层卷积网络
tags: [医学统计, 数据科学, 深度学习, 图像分析]
status: 已建
difficulty: advanced
question_type: 特征复用型深层图像建模
data_type: [图像数据, 三维影像数据, 栅格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [torch, torchvision]
r_packages: [keras]
---

# DenseNet（Densely Connected Convolutional Network, DenseNet）

## 1. 方法概览

### 1.1 定义

DenseNet 是一种密集连接卷积网络。它让每一层都接收所有前面层的特征图作为输入，从而强化特征复用并改善梯度传播。

### 1.2 它主要解决什么问题

- 研究问题：如何在保持深层表达能力的同时，更充分地复用浅层和中层特征。
- 适用任务：图像分类、迁移学习、医学影像表征提取。
- 常见医学场景：胸片识别、病理切片分类、皮肤和眼底图像分级。

### 1.3 直觉理解

DenseNet 像一套层层共享笔记的网络。后面每一层不仅看上一层结果，还能直接查看更早层的特征，因此对细粒度纹理和高层语义都能同时利用。

## 2. 数学形式

### 2.1 核心公式

DenseNet 的第 $l$ 层通常写为：

$$
x_l = H_l([x_0, x_1, \dots, x_{l-1}])
$$

其中 $[\cdot]$ 表示特征拼接，$H_l$ 是卷积、归一化和激活组成的变换。增长率 $k$ 表示每层新增的特征图数量。

### 2.2 参数或统计量含义

- 密集连接：当前层可直接访问所有前层特征。
- growth rate：每层新增通道数。
- transition layer：用于压缩通道和下采样。

### 2.3 关键假设

- 任务受益于多层级特征共享。
- 训练数据和计算资源足以支持密集连接带来的内存开销。
- 特征复用优于简单加深网络。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：二维或三维图像张量。
- 因变量类型：分类、回归、视觉识别任务标签。
- 数据结构：图像张量。
- 是否适合高维数据：适合图像高维输入，但内存开销需关注。
- 是否适合缺失较多数据：需先处理图像质量问题。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：若是图像序列，需额外时序建模。

### 3.2 示例表格

以肺结节影像分类为例：

| ImageID | ImagePath | SliceType | NoduleLabel |
| --- | --- | --- | --- |
| C001 | ct_001.png | axial | malignant |
| C002 | ct_002.png | axial | benign |
| C003 | ct_003.png | coronal | malignant |
| C004 | ct_004.png | axial | benign |

### 3.3 输入与产出

#### 输入

- 输入数据：图像张量和分类标签。
- 关键变量：网络版本、growth rate、压缩率、学习率。
- 需要预处理的内容：标准化、增强、中心和病人级划分。

#### 产出

- 模型对象/统计结果：DenseNet 权重和验证指标。
- 参数估计：各 dense block 与分类头参数。
- 预测结果：类别概率、回归值或图像嵌入。
- 不确定性指标：外部验证性能、校准与错误模式分析。

## 4. 适用场景

- 适合：希望充分复用多尺度特征的影像任务，尤其是迁移学习场景。
- 不适合：显存十分有限、样本特别少且无法迁移学习的任务。
- 使用前需要特别检查的点：内存占用、特征冗余、域偏移。

## 5. 实现

### 5.1 Python

常用包：

- `torch`
- `torchvision`

```python
from torchvision.models import densenet121

model = densenet121(weights=None, num_classes=2)
```

### 5.2 R

常用包：

- `keras`

```r
# R 中通常通过 keras 的 DenseNet 预训练应用模型做迁移学习，
# 或直接在 Python 生态中训练后导出结果。
```

## 6. 结果如何解释

- 核心结果看什么：分类性能、外部泛化、不同病灶类型下表现。
- 每个主要参数如何解释：growth rate 越大，新增特征越多，容量和资源需求也越高。
- 临床或医学意义如何表达：DenseNet 常用于提高视觉任务性能，但仍需临床验证和解释工具辅助。
- 常见误读：特征共享多不代表一定优于 ResNet，具体效果依赖任务和数据域。

## 7. 推荐可视化

- 训练/验证曲线。
- 混淆矩阵。
- 特征嵌入投影或热图解释。

## 8. 优势、局限与常见坑

### 优势

- 特征复用充分。
- 梯度传播顺畅。
- 在部分医疗影像任务中迁移效果优秀。

### 局限

- 内存占用较高。
- 结构较复杂。
- 解释性依然有限。

### 常见坑

- 忽略 dense block 带来的显存压力。
- 只比平均性能，不看罕见病灶类别。
- 不检查数据集来源偏差。

## 9. 与相近方法的区别

- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：DenseNet 是 CNN 的具体密集连接变体。
- 和 [[残差网络（Residual Network, ResNet）]] 的区别：ResNet 用逐块相加的残差连接，DenseNet 用跨层拼接复用特征。
- 和 [[AlexNet（AlexNet）]] 的区别：AlexNet 更早、更浅，DenseNet 更强调深层特征复用。

## 10. 医学研究中的典型应用

- 胸片和 CT 分类任务。
- 迁移学习驱动的病理图像识别。
- 作为医学影像表征骨干网络。

## 11. 相关方法

- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[残差网络（Residual Network, ResNet）]]
- [[AlexNet（AlexNet）]]
- [[Transformer（Transformer）]]

## 12. 参考资料

- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017:4700-4708.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- torchvision Developers. DenseNet models. [https://pytorch.org/vision/stable/models/densenet.html](https://pytorch.org/vision/stable/models/densenet.html) （访问日期：2026-07-02）

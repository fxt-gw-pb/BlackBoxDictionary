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

### 1.1 一句话本质

DenseNet 让每一层直接看到并保留所有前层特征，通过通道拼接实现特征复用和短梯度路径，而不是反复覆盖旧表示。

### 1.2 定义

DenseNet 是由 dense block 与 transition layer 组成的深层卷积架构。在一个 dense block 内，第 $\ell$ 层接收初始输入及前 $\ell-1$ 层输出的拼接；每层只新增少量特征图。transition layer 用 $1\times1$ 卷积和池化控制通道与空间尺寸。

### 1.3 它主要解决什么问题

- 研究问题：深网中早期细节被后续层反复变换、梯度路径过长时，如何让特征直接复用。
- 适用任务：图像分类、迁移学习、检测和分割骨干。
- 常见医学场景：胸片分类、病理切片分型、皮肤/眼底分级、CT/MRI 表征。

### 1.4 直觉与类比

ResNet 像把旧笔记与新修改相加；DenseNet 像把每次新笔记都追加到共享档案夹，后续层可随时翻阅所有旧页。每层无需重新发明边缘和纹理，只需贡献少量新特征。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

普通顺序 CNN 中，第十层只能通过第九层间接获取第一层信息；若中间变换丢掉细节，后层无法直接恢复。加深还让梯度必须穿过很长路径。DenseNet 为每一对前后层建立直接连接，让早期细粒度特征和监督梯度都走短路。

### 2.2 关键洞察

DenseNet 不让一层生成完整的新宽表示，而是每层只产生 $k$ 个新通道，再与已有通道拼接。$k$ 称 growth rate。网络深度增加时“档案夹”逐步变厚，但每层计算可保持窄小；bottleneck 与 compression 再控制成本。

### 2.3 与朴素/相邻做法的对比

- 相比普通 CNN：每层直接访问所有历史特征。
- 相比 ResNet：DenseNet 拼接而非相加，不会把旧特征与新特征压在同一通道。
- 相比 U-Net 跳连：U-Net 主要跨编码器—解码器尺度连接；DenseNet 在块内连接所有前层。
- 代价：拼接会增加通道和激活内存，连接数约随深度平方增长。

## 3. 数学形式

### 3.1 核心公式

第 $\ell$ 层输入为

$$
\mathbf x_\ell
=H_\ell\!\left(
[\mathbf x_0,\mathbf x_1,\ldots,\mathbf x_{\ell-1}]
\right)
$$

方括号表示沿通道维拼接。若初始通道数为 $k_0$，每层新增 $k$ 个通道，则经过 $L$ 层：

$$
k_{\text{out}}=k_0+Lk
$$

这个式子在说：旧特征原样保留，每层只把自己的 $k$ 个新特征追加进去。

DenseNet-BC 中一层常为

$$
H_\ell
=\operatorname{Conv}_{3\times3}
\circ\operatorname{BN}\circ\operatorname{ReLU}
\circ\operatorname{Conv}_{1\times1}
\circ\operatorname{BN}\circ\operatorname{ReLU}
$$

transition 的压缩因子 $\theta$ 将 $m$ 个通道降到 $\lfloor\theta m\rfloor$。

### 3.2 推导脉络

若希望第 $\ell$ 层复用所有早期特征，最直接方法就是拼接而不是反复相加/覆盖。为防止通道爆炸，让每层只输出较小 growth rate；在 $3\times3$ 卷积前用 $1\times1$ bottleneck 降低输入通道，并在 block 之间压缩和下采样。梯度可从损失直接到达任意早层，形成隐式深度监督。

### 3.3 参数与统计量含义

- $\mathbf x_0$：dense block 初始输入。
- $\mathbf x_\ell$：第 $\ell$ 层新增的特征图。
- $[\cdot]$：通道拼接，空间高宽必须一致。
- $k$：growth rate，每层新增通道数。
- $L$：block 内层数。
- bottleneck：用 $1\times1$ 卷积减少 $3\times3$ 卷积输入。
- compression $\theta$：transition 保留通道比例。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 多层特征值得同时保留 | 浅层细节与深层语义都有用 | 拼接只增加冗余 | 消融、通道利用分析 |
| 同一 block 空间尺寸一致 | 特征可沿通道拼接 | 形状不匹配 | 逐层形状日志 |
| growth rate 足够且不过大 | 每层有新增表达又不爆内存 | 欠拟合或内存过高 | 比较 $k$ 与学习曲线 |
| 训练/部署图像域相近 | 复用特征可迁移 | 外部性能下降 | 跨中心/设备验证 |
| 患者级切分正确 | 无同源图像泄漏 | 性能虚高 | patient_id、近重复审计 |

## 4. 手把手算例

先用标量“通道”展示拼接。初始特征

$$
\mathbf x_0=[1,2]
$$

设每层只新增一个通道，即 growth rate $k=1$。

**第 1 层：**

$$
x_1=H_1([1,2])=1+2=3
$$

拼接后为 $[1,2,3]$。

**第 2 层：**

$$
x_2=H_2([1,2,3])=1-2+3=2
$$

拼接后为 $[1,2,3,2]$。

**第 3 层：**

$$
x_3=H_3([1,2,3,2])=1+2-3+2=2
$$

最终输出

$$
[\mathbf x_0,x_1,x_2,x_3]=[1,2,3,2,2]
$$

初始两个特征始终原样可见，三层各新增一个特征，通道从 $k_0=2$ 增至

$$
k_{\text{out}}=2+3\times1=5
$$

再看真实卷积参数量：若 $k_0=3$、$k=2$，三层均直接用 $3\times3$ 卷积且不计偏置，则输入通道依次为 3、5、7：

$$
3\times2\times9+5\times2\times9+7\times2\times9
=54+90+126=270
$$

这也提醒：密集连接本身不保证每个玩具配置参数更少；真实 DenseNet 靠小 growth rate、bottleneck 和 compression 相对“宽层堆叠”提高效率。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：二维/三维医学图像与规则栅格。
- 因变量类型：分类、连续预测、检测或分割。
- 数据结构：保留空间与通道维的张量。
- 是否适合高维数据：适合，但激活内存可能较大。
- 是否适合缺失较多数据：不直接处理缺失模态。
- 是否适合删失数据：需接生存损失。
- 是否适合重复测量数据：按患者切分，时序关系另建模。

### 5.2 示例表格

| patient_id | image_path | view | site | label |
| --- | --- | --- | --- | ---: |
| P01 | p01_pa.png | PA | A | 0 |
| P02 | p02_ap.png | AP | B | 1 |

### 5.3 输入与产出

#### 输入

- 图像、标签、患者/设备元数据。
- block 配置、growth rate、compression 与预训练权重。
- 患者级数据划分和图像预处理。

#### 产出

- 分类概率或空间任务输出。
- 多尺度、多深度拼接特征。
- 训练历史与外部验证指标。
- 不自动产出可信因果解释或隐私保证。

## 6. 适用场景

- 适合：浅层纹理与深层语义都重要、希望特征复用、可利用预训练。
- 不适合：显存极受限、样本很小且无迁移学习、简单 CNN 已充分。
- 使用前检查：growth rate、激活内存、输入分辨率、患者级切分和域偏移。

## 7. 实现

### 7.1 Python

常用包：PyTorch、torchvision。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

dataset = datasets.FakeData(
    size=64, image_size=(3, 224, 224), num_classes=2,
    transform=transforms.ToTensor()
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
images, labels = next(iter(loader))
logits = model(images)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
print(logits.shape, float(loss))
```

### 7.2 R

常用包：keras3。下面实现一个最小 dense block。

```r
library(keras3)

inputs <- layer_input(shape = c(32, 32, 3))
h1 <- inputs |>
  layer_conv_2d(4, 3, padding = "same", activation = "relu")
x1 <- layer_concatenate(list(inputs, h1))

h2 <- x1 |>
  layer_conv_2d(4, 3, padding = "same", activation = "relu")
x2 <- layer_concatenate(list(x1, h2))

h3 <- x2 |>
  layer_conv_2d(4, 3, padding = "same", activation = "relu")
x3 <- layer_concatenate(list(x2, h3)) |>
  layer_global_average_pooling_2d()

outputs <- x3 |> layer_dense(2, activation = "softmax")
model <- keras_model(inputs, outputs)
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
summary(model)
```

## 8. 结果如何解读

- 最终按患者级外部性能、校准和阈值解释，不以连接数量判断优劣。
- DenseNet 的某层可直接使用早期特征，但不说明每个通道都有独立临床语义。
- 迁移学习需比较冻结、部分解冻和全量微调。
- 多切片预测需先按患者聚合。
- 热图和通道激活只用于行为诊断，不等于疾病机制。

## 9. 假设诊断与稳健性

- 比较 DenseNet、ResNet 和浅 CNN 的同切分性能。
- 记录各 block 通道数、显存和推理耗时。
- 改变 growth rate/compression 检查容量敏感性。
- 做患者级、检查级和近重复图像泄漏审计。
- 按中心、设备、视角和病灶大小分层验证。
- 多随机种子与预训练策略比较稳定性。

## 10. 推荐可视化

- dense block 的拼接连接图。
- 通道数随深度增长曲线。
- 不同 growth rate 的性能—显存图。
- 浅层/深层特征图和 Grad-CAM。
- 跨中心、设备与亚组性能图。

## 11. 优势、局限与常见坑

### 优势

- 直接复用所有前层特征，梯度路径短。
- 每层只需新增少量特征。
- 常适合作为迁移学习图像骨干。

### 局限

- 拼接导致通道和激活内存增长。
- 实现和部署复杂度高于简单顺序 CNN。
- 特征复用不自动带来解释性。

### 常见坑

- 把“连接密集”误写成“每层都很宽”。
- 混淆拼接与 ResNet 的逐元素相加。
- 忘记 transition 层压缩和下采样。
- 只按图像/切片随机切分。
- 因训练集表现好就忽略外部域偏移。

## 12. 与相近方法的区别

- 与 ResNet：DenseNet 拼接并保留各层身份；ResNet 相加形成修正后的同维表示。
- 与普通 CNN：DenseNet 每层直接访问所有先前输出。
- 与 U-Net：U-Net 跨尺度连接编码器与解码器，DenseNet 在 block 内密集连接。
- 与 AlexNet：DenseNet 更深、特征复用更充分，AlexNet 结构更简单。
- 如何选择：显存允许且细粒度特征复用可能重要时比较 DenseNet；通用稳健基线优先 ResNet。

## 13. 医学研究中的典型应用

- 胸片多标签异常识别。
- 病理、眼底和皮肤图像分类。
- CT/MRI 影像表征提取。
- 分割网络中的 dense block。

应报告患者级切分、预训练来源、growth rate/版本、设备和中心分布、类别不平衡及外部验证。普通分类 DenseNet 不处理删失。

## 14. 关键术语

- **密集连接（Dense connectivity）**：一层接收所有先前层输出。
- **拼接（Concatenation）**：沿通道维保留并排列多个特征张量。
- **Growth rate**：每层新增的特征通道数。
- **Dense block**：空间尺寸不变、内部密集连接的一组层。
- **Transition layer**：block 之间压缩通道和下采样的模块。
- **Compression**：transition 减少通道的比例。
- **Bottleneck**：用 $1\times1$ 卷积降低后续卷积成本。
- **特征复用（Feature reuse）**：后层直接使用早层已有特征。

## 15. 相关方法

- [[残差网络（Residual Network, ResNet）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[AlexNet（AlexNet）]]
- [[Transformer（Transformer）]]
- [[自编码器（Autoencoder）]]

## 16. 参考资料

- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. *CVPR*. 2017:4700-4708. https://doi.org/10.1109/CVPR.2017.243
- Pleiss G, Chen D, Huang G, Li T, van der Maaten L, Weinberger KQ. Memory-efficient implementation of DenseNets. arXiv:1707.06990; 2017.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- torchvision developers. DenseNet. https://pytorch.org/vision/stable/models/densenet.html

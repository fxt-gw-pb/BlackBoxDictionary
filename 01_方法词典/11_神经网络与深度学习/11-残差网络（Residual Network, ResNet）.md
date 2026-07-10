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

### 1.1 一句话本质

ResNet 不要求每组层从头学习完整映射，而是学习“在输入基础上还要改多少”，并用快捷连接给信息和梯度提供一条直接通路。

### 1.2 定义

残差网络是在深层卷积网络中堆叠残差块的架构。典型块输出 $\mathbf y=\mathcal F(\mathbf x)+\mathbf x$；当尺寸不同，则用投影快捷连接匹配。残差连接缓解的是深层网络的优化退化和梯度传播困难，不等于完全消除过拟合或梯度问题。

### 1.3 它主要解决什么问题

- 研究问题：网络加深后训练误差反而上升时，如何让新增层至少能方便地表示恒等映射。
- 适用任务：图像分类、检测、分割、迁移学习和特征提取。
- 常见医学场景：CT/MRI 分类、病理切片分型、胸片识别、医学影像骨干网络。

### 1.4 直觉与类比

普通层像要求医生重写整份病历摘要；残差块只要求写“相对上一版的修改意见”。若没有需要修改的内容，让残差接近 0 就能原样传递；快捷连接还像一条不经过复杂审批的直达通道，让早期信号和梯度不必穿过所有卷积。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

理论上，给一个表现良好的浅层网络再加几层，新层只要学习恒等映射，性能不应变差；实践中普通深网却可能出现训练误差上升的“退化问题”。这说明障碍不只是过拟合，而是优化器很难让多层非线性恰好合成恒等映射。

### 2.2 关键洞察

把目标映射 $\mathcal H(\mathbf x)$ 改写为

$$
\mathcal F(\mathbf x)=\mathcal H(\mathbf x)-\mathbf x
$$

于是只需学习残差 $\mathcal F$。若理想映射接近恒等，残差接近 0 比多层网络直接拟合 $\mathcal H(\mathbf x)=\mathbf x$ 更容易。同时

$$
\frac{\partial\mathbf y}{\partial\mathbf x}
=\mathbf I+\frac{\partial\mathcal F}{\partial\mathbf x}
$$

恒等项让梯度拥有直接分量。

### 2.3 与朴素/相邻做法的对比

- 相比普通 CNN：ResNet 通过加法快捷连接重用输入。
- 相比 DenseNet：ResNet 做逐元素相加，通道数通常不累积；DenseNet 拼接所有旧特征。
- 相比 Highway Network：ResNet 不依赖额外门控即可建立恒等通路。
- 相比简单加深：残差连接改善可训练性，但仍需规范化、初始化和正则化。

## 3. 数学形式

### 3.1 核心公式

基本残差块为

$$
\mathbf y=\phi\!\left(\mathcal F(\mathbf x;\mathbf W)+\mathbf x\right)
$$

若空间尺寸或通道数变化：

$$
\mathbf y=\phi\!\left(
\mathcal F(\mathbf x;\mathbf W)+\mathbf W_s\mathbf x
\right)
$$

这个式子在说：主分支学习修正量，快捷分支传递原输入；两者形状一致后逐元素相加。

两层基本块可写为

$$
\mathcal F(\mathbf x)
=\mathbf W_2\,\phi(\operatorname{BN}(\mathbf W_1\mathbf x))
$$

深层 ResNet 常用 $1\times1$、$3\times3$、$1\times1$ 的 bottleneck 块降低计算。

### 3.2 推导脉络

若最优新增模块什么也不做，普通深层块需要多个权重共同实现恒等映射；残差块只需把主分支权重学到近 0。反向传播时，损失梯度可分成快捷分支的恒等项与残差分支项，降低全部依赖长矩阵连乘的程度。投影快捷连接只在形状变化时使用，否则恒等连接最直接。

### 3.3 参数与统计量含义

- $\mathbf x$：残差块输入。
- $\mathcal F$：由卷积、归一化和激活组成的残差分支。
- $\mathbf W_s$：匹配尺寸的投影，常为 $1\times1$ 卷积。
- $\phi$：块输出激活；pre-activation ResNet 将归一化/激活移到卷积前。
- stage：保持同一空间尺度的一组残差块。
- depth：网络层数，不能直接等同于有效复杂度或临床性能。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 任务受益于深层层级特征 | 更大感受野和抽象层有增益 | 深网只增加方差和成本 | 与浅 CNN 比较学习曲线 |
| 快捷分支形状匹配 | 加法两端维度一致 | 无法计算或语义错位 | 逐层记录张量形状 |
| 图像域与预训练域可迁移 | 低层特征可复用 | 迁移收益有限或负迁移 | 冻结/微调对照 |
| 患者级切分无泄漏 | 同一患者图像不跨集合 | 性能虚高 | patient_id 与近重复审计 |
| 批归一化统计稳定 | 批量足够且部署域相近 | 小批量/域偏移不稳定 | 比较 GroupNorm、冻结 BN |

## 4. 手把手算例

用二维线性残差块说明前向与梯度直通。设

$$
\mathbf x=(2,1)^\top,\qquad
\mathcal F(\mathbf x)=\mathbf W\mathbf x,\qquad
\mathbf W=
\begin{pmatrix}
0.1&0\\
0&-0.2
\end{pmatrix}
$$

**第 1 步：主分支只学修正量。**

$$
\mathcal F(\mathbf x)=(0.2,-0.2)^\top
$$

**第 2 步：快捷连接相加。**

$$
\mathbf y=\mathbf x+\mathcal F(\mathbf x)
=(2.2,0.8)^\top
$$

输入主体被保留，只做小幅调整。若没有快捷连接，普通块输出仅为 $(0.2,-0.2)$。

**第 3 步：看梯度。** 令损失 $L=y_1+y_2$，则

$$
\frac{\partial L}{\partial\mathbf x}
=(1,1)(\mathbf I+\mathbf W)
=(1.1,0.8)
$$

没有快捷连接时梯度仅为

$$
(1,1)\mathbf W=(0.1,-0.2)
$$

恒等通路贡献了 $(1,1)$，使梯度不必完全依赖残差权重。这就是残差连接帮助深层优化的数值本质。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：二维图像、三维体数据或规则栅格。
- 因变量类型：分类、连续结局、分割或检测标签。
- 数据结构：保留空间和通道维的张量。
- 是否适合高维数据：适合，但显存和有效样本量是限制。
- 是否适合缺失较多数据：不直接处理缺失模态/切片。
- 是否适合删失数据：需接专门生存损失。
- 是否适合重复测量数据：图像需按患者切分，时序相关另建模。

### 5.2 示例表格

| patient_id | image_path | modality | scanner | label |
| --- | --- | --- | --- | ---: |
| P01 | p01_ct.nii.gz | CT | A | 0 |
| P02 | p02_ct.nii.gz | CT | B | 1 |

### 5.3 输入与产出

#### 输入

- 规范化图像、标签与患者/设备元数据。
- ResNet 深度、输入通道、预训练权重和微调策略。
- 患者级训练/验证/测试划分。

#### 产出

- 类别概率、连续预测或空间任务输出。
- 全局池化前后的图像表征。
- 训练历史与验证指标。
- 不自动产出可信机制解释或不确定性。

## 6. 适用场景

- 适合：需要成熟图像骨干、迁移学习或较深层级特征。
- 不适合：样本极少且无预训练、二维投影丢失关键三维信息、简单模型已充分。
- 使用前检查：输入尺寸/通道、患者级切分、设备偏倚、BN、小病灶与外部验证。

## 7. 实现

### 7.1 Python

常用包：PyTorch、torchvision。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

dataset = datasets.FakeData(
    size=128, image_size=(3, 224, 224), num_classes=2,
    transform=transforms.ToTensor()
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

images, labels = next(iter(loader))
optimizer.zero_grad()
loss = loss_fn(model(images), labels)
loss.backward()
optimizer.step()
print(float(loss))
```

真实医学数据应换成按患者划分的数据集；迁移学习时可使用预训练权重并比较冻结与全量微调。

### 7.2 R

常用包：keras3。

```r
library(keras3)

inputs <- layer_input(shape = c(32, 32, 3))
x <- inputs |>
  layer_conv_2d(16, 3, padding = "same", activation = "relu")

residual <- x |>
  layer_conv_2d(16, 3, padding = "same", activation = "relu") |>
  layer_conv_2d(16, 3, padding = "same")

x <- layer_add(list(x, residual)) |>
  layer_activation("relu") |>
  layer_global_average_pooling_2d()
outputs <- x |> layer_dense(2, activation = "softmax")

model <- keras_model(inputs, outputs)
model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
summary(model)
```

## 8. 结果如何解读

- ResNet 是特征提取架构；结果仍需以患者级外部 AUC/PR-AUC、校准和阈值性能解释。
- 更深网络验证性能更好才有意义，层数本身不是证据。
- 预训练收益可能来自通用纹理，也可能受自然图像与医学域差异限制。
- Grad-CAM 只作行为诊断，不证明病灶区域是因果依据。
- 多切片/多视角预测需先按患者聚合再评价。

## 9. 假设诊断与稳健性

- 比较 ResNet-18/34/50 与浅 CNN，检查加深是否真有收益。
- 观察训练/验证曲线和残差分支输出范数。
- 小批量训练比较 BatchNorm、GroupNorm 或冻结统计。
- 做图像近重复、患者级与检查级泄漏审计。
- 按设备、中心、视角和病灶大小分层评价。
- 多随机种子与不同预训练/微调策略做敏感性分析。

## 10. 推荐可视化

- 残差块主分支与快捷分支结构图。
- 各 stage 的张量尺寸和通道数。
- 训练/验证曲线与不同深度性能曲线。
- Grad-CAM/遮挡图及伪影对照。
- 跨中心、设备和病灶大小性能森林图。

## 11. 优势、局限与常见坑

### 优势

- 恒等快捷连接让很深 CNN 更易优化。
- 架构成熟，预训练模型和下游生态丰富。
- 可作为分类、检测和分割的通用骨干。

### 局限

- 深度增加计算、显存和过拟合风险。
- 加法融合要求形状一致，可能混合语义不同特征。
- 不自动解决域偏移、泄漏或解释问题。

### 常见坑

- 将退化问题误写成普通过拟合。
- 尺寸变化时忘记投影快捷连接。
- 小批量仍直接更新 BatchNorm 统计。
- 只按切片随机切分。
- 不经验证就认为更深一定更好。

## 12. 与相近方法的区别

- 与普通 CNN：ResNet 显式学习残差并保留恒等通路。
- 与 DenseNet：ResNet 用加法融合；DenseNet 用通道拼接保留所有旧特征。
- 与 AlexNet：AlexNet 是早期顺序堆叠架构，ResNet 可训练得更深。
- 与 Vision Transformer：ResNet 强局部先验、样本效率较好；ViT 更直接建模全局关系。
- 如何选择：中小型医学影像任务先用预训练 ResNet 建强基线，再比较 DenseNet/ViT。

## 13. 医学研究中的典型应用

- 胸片、眼底和皮肤镜图像分类。
- CT/MRI 肿瘤分型与预后表征。
- 病理图像块编码及多实例聚合。
- 分割/检测模型的编码器骨干。

应报告患者级切分、图像预处理、预训练来源、设备/中心分布、类别不平衡和外部验证。时间到事件结局需另接删失感知目标。

## 14. 关键术语

- **残差（Residual）**：目标映射相对输入的修正量。
- **快捷连接（Shortcut connection）**：绕过主分支直接传递输入的路径。
- **恒等映射（Identity mapping）**：输出等于输入的映射。
- **退化问题（Degradation problem）**：网络加深后训练误差反而升高。
- **投影快捷连接（Projection shortcut）**：用 $1\times1$ 卷积匹配形状。
- **Bottleneck block**：用 $1\times1$ 卷积先降维、再升维的残差块。
- **Pre-activation**：归一化和激活置于卷积之前的残差设计。
- **Stage**：共享空间尺度和通道设定的一组残差块。

## 15. 相关方法

- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[DenseNet（Densely Connected Convolutional Network, DenseNet）]]
- [[AlexNet（AlexNet）]]
- [[Transformer（Transformer）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]

## 16. 参考资料

- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *CVPR*. 2016:770-778. https://doi.org/10.1109/CVPR.2016.90
- He K, Zhang X, Ren S, Sun J. Identity mappings in deep residual networks. *ECCV*. 2016:630-645. https://doi.org/10.1007/978-3-319-46493-0_38
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- torchvision developers. ResNet. https://pytorch.org/vision/stable/models/resnet.html

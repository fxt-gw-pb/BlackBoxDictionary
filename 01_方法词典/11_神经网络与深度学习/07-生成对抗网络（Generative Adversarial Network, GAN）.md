---
title: 生成对抗网络
english_name: Generative Adversarial Network, GAN
slug: generative-adversarial-network-gan
aliases: [GAN, generative adversarial network, "生成对抗网络（Generative Adversarial Network, GAN）"]
category: 神经网络与深度学习
subcategory: 生成式神经网络
tags: [医学统计, 数据科学, 深度学习, 生成模型]
status: 已建
difficulty: advanced
question_type: 数据生成与分布拟合
data_type: [图像数据, 高维特征矩阵, 序列数据]
outcome_type: [生成样本, 表征学习]
python_packages: [torch, tensorflow, keras]
r_packages: [keras]
---

# 生成对抗网络（Generative Adversarial Network, GAN）

## 1. 方法概览

### 1.1 定义

GAN 是由生成器和判别器组成的生成模型。生成器尝试生成逼真的伪样本，判别器尝试区分真实样本和伪样本，两者在对抗训练中共同逼近真实数据分布。

### 1.2 它主要解决什么问题

- 研究问题：如何在没有显式概率密度公式的情况下学习复杂高维数据分布并生成新样本。
- 适用任务：图像生成、数据增强、缺失模态补全、域适配、隐私保护合成数据。
- 常见医学场景：小样本医学影像增强、病理图像风格迁移、罕见病例合成。

### 1.3 直觉理解

可以把 GAN 理解成“造假者”和“审稿人”的博弈。造假者不断改进伪造样本，审稿人不断提高识别能力，直到伪样本足够像真样本。

## 2. 数学形式

### 2.1 核心公式

经典 GAN 的目标函数为：

$$
\min_G \max_D \; V(D,G)
= \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

其中 $G(z)$ 为生成样本，$D(x)$ 为样本来自真实分布的概率。

### 2.2 参数或统计量含义

- $G$：生成器，将随机噪声映射为伪样本。
- $D$：判别器，区分真实与生成样本。
- $z$：潜在噪声向量。
- 对抗损失：衡量生成器和判别器的博弈过程。

### 2.3 关键假设

- 训练数据足以代表目标分布。
- 生成器和判别器容量相对平衡。
- 训练过程能在不稳定对抗中维持可接受收敛。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：图像、表征向量、序列等高维样本。
- 因变量类型：通常是无监督生成目标，也可扩展到条件标签。
- 数据结构：真实样本张量加潜在噪声分布。
- 是否适合高维数据：设计初衷就是高维生成，但训练难度也随之上升。
- 是否适合缺失较多数据：不直接针对缺失机制，需专门设计。
- 是否适合删失数据：不适合直接做删失分析。
- 是否适合重复测量数据：可扩展到序列 GAN，但需专门结构。

### 3.2 示例表格

以病理图像增强为例：

| SampleID | ImagePatch | TissueType | RarePattern |
| --- | --- | --- | --- |
| S001 | patch_001.png | breast | 1 |
| S002 | patch_002.png | breast | 0 |
| S003 | patch_003.png | colon | 1 |
| S004 | patch_004.png | colon | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：真实样本和随机噪声向量。
- 关键变量：潜在维度、生成器/判别器结构、学习率、判别器训练步数。
- 需要预处理的内容：标准化、类别平衡、生成目标定义、稳定训练策略。

#### 产出

- 模型对象/统计结果：训练好的生成器和判别器。
- 参数估计：两个网络的权重。
- 预测结果：生成样本、潜在表示或条件生成结果。
- 不确定性指标：生成质量评估、下游任务提升效果、人工或专家质控。

## 4. 适用场景

- 适合：需要生成逼真样本或做数据增强的任务，尤其是图像领域。
- 不适合：样本极少、对稳定概率估计要求高、需要严格可解释机制的任务。
- 使用前需要特别检查的点：模式崩溃、隐私泄露风险、生成样本是否引入伪特征。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch.nn as nn

generator = nn.Sequential(
    nn.Linear(100, 256), nn.ReLU(),
    nn.Linear(256, 784), nn.Tanh()
)

discriminator = nn.Sequential(
    nn.Linear(784, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1), nn.Sigmoid()
)
```

### 5.2 R

常用包：

- `keras`

```r
# R 中通常通过 keras 自定义 generator 和 discriminator，
# 再手工编写交替训练循环来实现 GAN。
```

## 6. 结果如何解释

- 核心结果看什么：生成样本质量、多样性、下游任务增益、是否出现模式崩溃。
- 每个主要参数如何解释：潜在维度控制生成自由度，判别器强弱影响训练平衡。
- 临床或医学意义如何表达：生成样本只能作为增强或研究工具，不能替代真实临床证据。
- 常见误读：生成图像“看起来真”不等于包含真实生物学机制，也不代表适合直接用于诊断。

## 7. 推荐可视化

- 不同训练轮次的生成样本网格。
- 生成器和判别器损失曲线。
- 真实样本与生成样本的特征嵌入图。

## 8. 优势、局限与常见坑

### 优势

- 擅长生成视觉质量较高的样本。
- 可用于数据增强和域适配。
- 不需要显式写出高维分布形式。

### 局限

- 训练不稳定。
- 缺少统一、可靠的生成质量评估标准。
- 存在模式崩溃和隐私泄露风险。

### 常见坑

- 只看生成样本好不好看，不检验下游价值。
- 用生成样本替代真实外部验证数据。
- 忽略 rare class 是否真的被有效增强。

## 9. 与相近方法的区别

- 和 [[自编码器（Autoencoder）]] 的区别：自编码器强调重构和压缩，GAN 强调逼真生成。
- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：CNN 是网络结构家族，GAN 是一种生成式训练框架。
- 和 [[Transformer（Transformer）]] 的区别：Transformer 更常作为序列建模骨架，GAN 是对抗式生成机制。

## 10. 医学研究中的典型应用

- 医学影像稀有类别增强。
- 病理图像风格迁移与域对齐。
- 合成样本用于算法预训练或隐私保护研究。

## 11. 相关方法

- [[自编码器（Autoencoder）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[Transformer（Transformer）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]

## 12. 参考资料

- Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets. *Advances in Neural Information Processing Systems*. 2014;27.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- Creswell A, White T, Dumoulin V, et al. Generative adversarial networks: an overview. *IEEE Signal Processing Magazine*. 2018;35(1):53-65.

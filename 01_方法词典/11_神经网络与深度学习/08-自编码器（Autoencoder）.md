---
title: 自编码器
english_name: Autoencoder
slug: autoencoder
aliases: [AE, autoencoder, "自编码器（Autoencoder）"]
category: 神经网络与深度学习
subcategory: 表征学习与重构
tags: [医学统计, 数据科学, 深度学习, 无监督学习]
status: 已建
difficulty: intermediate
question_type: 无监督表征学习与重构
data_type: [表格数据, 图像数据, 高维特征矩阵]
outcome_type: [重构输出, 低维表征, 异常分数]
python_packages: [tensorflow, keras, torch]
r_packages: [keras]
---

# 自编码器（Autoencoder）

## 1. 方法概览

### 1.1 定义

自编码器是一类无监督或自监督神经网络。它通过编码器把输入压缩为低维潜在表示，再通过解码器尽量重构原始输入，从而学习有信息量的表征。

### 1.2 它主要解决什么问题

- 研究问题：如何在没有人工标签的情况下学习数据表示，并实现降维、去噪、压缩或异常检测。
- 适用任务：特征压缩、去噪重构、异常检测、表示学习。
- 常见医学场景：病理图像压缩表征、基因表达降维、监测数据异常识别、缺损图像重建。

### 1.3 直觉理解

自编码器像一个先“压缩再解压”的模型。如果压缩后的信息仍能很好还原输入，说明中间的低维表示抓住了数据中的关键结构。

## 2. 数学形式

### 2.1 核心公式

编码器和解码器可写为：

$$
z = f_{\theta}(x)
$$

$$
\hat x = g_{\phi}(z)
$$

训练目标常为最小化重构误差：

$$
\min_{\theta,\phi} \sum_{i=1}^{n} \mathcal{L}(x_i, \hat x_i)
$$

其中 $\mathcal{L}$ 常取均方误差或二元交叉熵。

### 2.2 参数或统计量含义

- 编码器：把原始输入压缩为潜在变量。
- 解码器：从潜在变量重构输入。
- bottleneck 维度：控制压缩强度和表征容量。
- 重构误差：衡量模型保留原始信息的程度。

### 2.3 关键假设

- 数据中存在可压缩的潜在结构。
- 重构任务能诱导出对下游任务有用的表示。
- 噪声和异常模式与主体分布存在差异。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征矩阵、图像、信号。
- 因变量类型：通常无显式标签。
- 数据结构：向量或张量输入。
- 是否适合高维数据：适合高维数据压缩和表示学习。
- 是否适合缺失较多数据：需专门设计掩码或去噪目标。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：若是序列型数据，需结合时序编码器。

### 3.2 示例表格

以实验室多指标异常检测为例：

| PatientID | ALT | AST | ALB | Creatinine | Urea |
| --- | --- | --- | --- | --- | --- |
| P101 | 35 | 31 | 42 | 0.9 | 4.7 |
| P102 | 168 | 143 | 32 | 2.1 | 13.5 |
| P103 | 29 | 25 | 44 | 0.8 | 4.3 |
| P104 | 41 | 39 | 40 | 1.0 | 5.0 |

### 3.3 输入与产出

#### 输入

- 输入数据：原始特征矩阵或图像张量。
- 关键变量：瓶颈维度、层数、损失函数、正则化方式。
- 需要预处理的内容：标准化、归一化、异常值和缺失处理。

#### 产出

- 模型对象/统计结果：编码器、解码器和训练损失。
- 参数估计：网络权重与偏置。
- 预测结果：低维潜在表示、重构值、重构误差。
- 不确定性指标：异常检测阈值稳定性、下游任务验证表现。

## 4. 适用场景

- 适合：高维无监督表征学习、数据去噪、输入压缩和异常检测。
- 不适合：需要明确概率解释或强因果解释的任务。
- 使用前需要特别检查的点：瓶颈是否过窄、重构是否学到无关细节、异常阈值如何定义。

## 5. 实现

### 5.1 Python

常用包：

- `tensorflow`
- `keras`

```python
from tensorflow import keras

inputs = keras.Input(shape=(100,))
encoded = keras.layers.Dense(32, activation="relu")(inputs)
decoded = keras.layers.Dense(100, activation="linear")(encoded)
autoencoder = keras.Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

inputs <- layer_input(shape = c(100))
encoded <- inputs |> layer_dense(units = 32, activation = "relu")
decoded <- encoded |> layer_dense(units = 100, activation = "linear")
autoencoder <- keras_model(inputs = inputs, outputs = decoded)
autoencoder |> compile(optimizer = "adam", loss = "mse")
```

## 6. 结果如何解释

- 核心结果看什么：重构误差、潜在空间可分性、下游任务效果。
- 每个主要参数如何解释：瓶颈维度越小，压缩越强，但可能丢失重要信息。
- 临床或医学意义如何表达：潜在表示可作为后续分型或预测输入，但不应直接当作临床机制解释。
- 常见误读：重构得好不代表表示一定适合所有下游任务。

## 7. 推荐可视化

- 原始输入与重构输入对比图。
- 潜在空间二维投影图。
- 重构误差分布图。

## 8. 优势、局限与常见坑

### 优势

- 无需人工标签即可学习表示。
- 可用于降维、去噪和异常检测。
- 易与卷积、时序等结构结合。

### 局限

- 表示质量高度依赖任务设计。
- 可解释性通常弱。
- 容易学到“复制输入”的捷径。

### 常见坑

- 瓶颈太宽，导致压缩约束不足。
- 只看重构损失，不验证下游价值。
- 用训练集重构误差直接定义异常阈值而不做独立验证。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 是线性降维，自编码器可学习非线性表征。
- 和 [[生成对抗网络（Generative Adversarial Network, GAN）]] 的区别：自编码器重在重构与表征，GAN 重在逼真生成。
- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：CNN 是结构形式，自编码器是一类训练目标和框架。

## 10. 医学研究中的典型应用

- 基因表达或代谢组学特征压缩。
- 医学影像去噪和重建。
- 监测数据异常检测和患者亚型表征。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[生成对抗网络（Generative Adversarial Network, GAN）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]

## 12. 参考资料

- Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. *Science*. 2006;313(5786):504-507.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- Chollet F. *Deep Learning with Python*. 2nd ed. Manning; 2021.

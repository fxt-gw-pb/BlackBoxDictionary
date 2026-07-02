---
title: 人工神经网络
english_name: Artificial Neural Network, ANN
slug: artificial-neural-network-ann
aliases: [ANN, artificial neural network, "人工神经网络（Artificial Neural Network, ANN）"]
category: 神经网络与深度学习
subcategory: 神经网络基础
tags: [医学统计, 数据科学, 神经网络, 深度学习]
status: 已建
difficulty: basic
question_type: 通用非线性映射建模
data_type: [表格数据, 图像向量, 序列数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [tensorflow, keras, torch]
r_packages: [keras, torch]
---

# 人工神经网络（Artificial Neural Network, ANN）

## 1. 方法概览

### 1.1 定义

人工神经网络是由多层神经元节点组成的一类函数逼近模型。它通过权重、偏置和非线性激活函数，把输入数据映射到分类、回归或表征学习目标上。

### 1.2 它主要解决什么问题

- 研究问题：当变量之间关系复杂、存在明显非线性或交互效应时，如何学习灵活的输入到输出映射。
- 适用任务：分类、回归、表示学习、图像识别、文本建模、序列预测。
- 常见医学场景：基于 EHR 的风险预测、影像识别、病理切片分类、多模态特征融合。

### 1.3 直觉理解

ANN 可以理解成一组逐层加工的特征变换器。前层先从原始输入中提取简单模式，中间层组合出更高阶表示，最后一层再根据这些表示输出预测结果。

## 2. 数学形式

### 2.1 核心公式

对第 $l$ 层网络，有：

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = \phi^{[l]}(z^{[l]})
$$

其中 $a^{[0]} = x$ 为输入。对整个训练集，常见优化目标为：

$$
\min_{\theta} \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(f_\theta(x_i), y_i)
$$

参数 $\theta$ 通过反向传播和梯度下降类算法更新。

### 2.2 参数或统计量含义

- $W^{[l]}$：第 $l$ 层权重矩阵。
- $b^{[l]}$：第 $l$ 层偏置向量。
- $\phi^{[l]}$：激活函数，如 ReLU、sigmoid、tanh。
- $\mathcal{L}$：损失函数，如交叉熵或均方误差。
- 学习率、层数、隐藏单元数：决定网络容量和训练稳定性。

### 2.3 关键假设

- 训练数据足以支持高容量模型学习。
- 输入特征经过合理数值化和预处理。
- 训练集与部署场景数据分布相近。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：数值特征、图像像素、嵌入向量、序列特征。
- 因变量类型：连续型、二分类、多分类或结构化输出。
- 数据结构：向量、矩阵、张量。
- 是否适合高维数据：适合，但更依赖样本量和正则化。
- 是否适合缺失较多数据：通常需先插补或显式建模缺失。
- 是否适合删失数据：标准 ANN 不直接处理删失机制。
- 是否适合重复测量数据：需结合 CNN、RNN、Transformer 等专门结构。

### 3.2 示例表格

以住院患者风险分类为例：

| Age | BMI | SBP | Creatinine | Hemoglobin | Outcome |
| --- | --- | --- | --- | --- | --- |
| 71 | 28.2 | 96 | 1.8 | 10.9 | high |
| 45 | 23.1 | 128 | 0.9 | 13.7 | low |
| 63 | 31.4 | 102 | 1.5 | 11.2 | high |
| 38 | 21.7 | 122 | 0.8 | 14.0 | low |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵或张量与目标标签。
- 关键变量：网络深度、隐藏单元数、激活函数、优化器、正则化。
- 需要预处理的内容：标准化、类别编码、训练验证划分、数据增强。

#### 产出

- 模型对象/统计结果：训练好的网络参数、损失曲线、验证性能。
- 参数估计：各层权重与偏置。
- 预测结果：类别概率、连续预测值或中间表征。
- 不确定性指标：验证集指标、校准结果、外部验证表现。

## 4. 适用场景

- 适合：存在明显非线性关系、特征维度较高、传统线性模型表达不足的任务。
- 不适合：极小样本、强解释性优先、简单规则即可解决的问题。
- 使用前需要特别检查的点：样本量、过拟合风险、数据泄露、外部可迁移性。

## 5. 实现

### 5.1 Python

常用包：

- `tensorflow`
- `keras`

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(5,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu", input_shape = c(5)) |>
  layer_dense(units = 16, activation = "relu") |>
  layer_dense(units = 3, activation = "softmax")

model |>
  compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
```

## 6. 结果如何解释

- 核心结果看什么：验证集性能、训练与验证损失差距、校准与泛化能力。
- 每个主要参数如何解释：层数和宽度控制表达能力，正则化和 dropout 控制复杂度。
- 临床或医学意义如何表达：通常强调预测能力和模式识别能力，而非单一参数的方向性解释。
- 常见误读：ANN 性能高不代表一定稳健，仍需时间外、中心外和人群外验证。

## 7. 推荐可视化

- 训练/验证损失曲线。
- ROC 曲线或 PR 曲线。
- 混淆矩阵或校准曲线。

## 8. 优势、局限与常见坑

### 优势

- 能拟合复杂非线性关系。
- 可扩展到图像、文本、序列和多模态任务。
- 通过不同网络结构可以适配多种数据形态。

### 局限

- 需要较多数据与计算资源。
- 参数可解释性通常较弱。
- 对训练细节和超参数较敏感。

### 常见坑

- 未划分独立验证集就反复调参。
- 类别不平衡时只关注总体准确率。
- 直接把训练集表现当成临床可用证据。

## 9. 与相近方法的区别

- 和 [[感知机（Perceptron）]] 的区别：ANN 是多层、多神经元的通用框架，感知机是最简单的单层线性特例。
- 和 [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]] 的区别：MLP 是最典型的前馈 ANN 具体实现。
- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：CNN 是面向局部空间结构数据的 ANN 变体。

## 10. 医学研究中的典型应用

- EHR 结构化数据风险预测。
- 影像和病理切片识别。
- 生理信号与文本信息的联合建模。

## 11. 相关方法

- [[感知机（Perceptron）]]
- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[循环神经网络（Recurrent Neural Network, RNN）]]

## 12. 参考资料

- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- Bishop CM. *Pattern Recognition and Machine Learning*. Springer; 2006.
- Chollet F. *Deep Learning with Python*. 2nd ed. Manning; 2021.

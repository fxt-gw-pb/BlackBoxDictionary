---
title: 门控循环单元
english_name: Gated Recurrent Unit, GRU
slug: gated-recurrent-unit-gru
aliases: [GRU, gated recurrent unit, "门控循环单元（Gated Recurrent Unit, GRU）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 高效门控序列建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 门控循环单元（Gated Recurrent Unit, GRU）

## 1. 方法概览

### 1.1 定义

GRU 是一种门控循环神经网络结构。它将 LSTM 的门控机制做了更紧凑的简化，通过更新门和重置门控制历史信息保留与新信息写入。

### 1.2 它主要解决什么问题

- 研究问题：如何在保留时序依赖建模能力的同时，减少参数量和训练复杂度。
- 适用任务：时间序列预测、序列分类、文本建模、事件序列分析。
- 常见医学场景：生命体征预警、药物使用序列预测、临床文本编码。

### 1.3 直觉理解

GRU 仍然是“会记忆的序列模型”，但它把 LSTM 的多个门合并得更简洁。这样它常能在较少参数下获得与 LSTM 接近的效果。

## 2. 数学形式

### 2.1 核心公式

GRU 的更新常写为：

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde h_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$

$$
h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde h_t
$$

### 2.2 参数或统计量含义

- $z_t$：更新门，控制新旧信息混合比例。
- $r_t$：重置门，控制历史状态参与候选状态计算的程度。
- $\tilde h_t$：候选隐藏状态。
- $h_t$：当前隐藏状态。

### 2.3 关键假设

- 顺序依赖对任务有价值。
- 需要比基础 RNN 更稳定的训练，但希望结构比 LSTM 更轻量。
- 训练样本与部署数据的时序模式相似。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单变量或多变量时间序列、token 序列、事件流。
- 因变量类型：分类标签、连续值、下一个事件或序列输出。
- 数据结构：三维序列张量。
- 是否适合高维数据：可以，但需要正则化与合适的样本量。
- 是否适合缺失较多数据：需先处理缺失或增加缺失指示。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：适合有顺序的重复测量数据。

### 3.2 示例表格

以住院患者夜间恶化预测为例：

| PatientID | Step | HR | SBP | RespRate | Deterioration |
| --- | --- | --- | --- | --- | --- |
| P010 | 1 | 88 | 114 | 18 | 0 |
| P010 | 2 | 93 | 109 | 20 | 0 |
| P010 | 3 | 104 | 98 | 24 | 1 |
| P011 | 1 | 74 | 126 | 15 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：按时间组织的序列窗口。
- 关键变量：隐藏单元数、层数、dropout、窗口长度。
- 需要预处理的内容：排序、标准化、时间切分、类别不平衡处理。

#### 产出

- 模型对象/统计结果：GRU 参数和验证集性能。
- 参数估计：更新门、重置门和输出层权重。
- 预测结果：类别概率、连续值或序列表示。
- 不确定性指标：验证误差、时间外泛化、分层表现。

## 4. 适用场景

- 适合：需要门控序列建模、但又希望减少参数和训练负担的场景。
- 不适合：极长序列且需要显式全局依赖建模时。
- 使用前需要特别检查的点：序列泄露、模型容量、训练稳定性。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch
from torch import nn

class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_classes=2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_gru(units = 64, input_shape = c(seq_len, n_features)) |>
  layer_dense(units = 2, activation = "softmax")
```

## 6. 结果如何解释

- 核心结果看什么：验证集误差、不同序列长度下性能、部署场景稳定性。
- 每个主要参数如何解释：隐藏单元数控制容量，窗口长度控制可见上下文。
- 临床或医学意义如何表达：更适合描述动态风险识别能力，而非单个参数临床效应值。
- 常见误读：GRU 比 LSTM 更轻量，不代表它在所有任务上都一定更好。

## 7. 推荐可视化

- 预测风险随时间变化曲线。
- 损失曲线。
- 与 RNN/LSTM 的性能对比图。

## 8. 优势、局限与常见坑

### 优势

- 结构简洁，参数较少。
- 常比普通 RNN 更稳定。
- 在中等规模序列任务中常有较好性价比。

### 局限

- 仍然是递归结构，并行效率有限。
- 解释性弱。
- 超参数和数据切分仍然敏感。

### 常见坑

- 不按患者或时间划分训练测试集。
- 以为参数更少就不需要正则化。
- 忽略部署时序漂移与数据缺失机制。

## 9. 与相近方法的区别

- 和 [[循环神经网络（Recurrent Neural Network, RNN）]] 的区别：GRU 有门控机制，能更稳定地保留历史信息。
- 和 [[长短期记忆网络（Long Short-Term Memory, LSTM）]] 的区别：GRU 无独立细胞状态，结构更简洁。
- 和 [[Transformer（Transformer）]] 的区别：GRU 按顺序递归处理，Transformer 依赖注意力并行建模全局关系。

## 10. 医学研究中的典型应用

- 动态监测序列的短期预警。
- 临床事件流序列建模。
- 医疗文本与编码序列分析。

## 11. 相关方法

- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[Transformer（Transformer）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]

## 12. 参考资料

- Cho K, van Merrienboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv*. 2014. [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- PyTorch Developers. `torch.nn.GRU`. [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) （访问日期：2026-07-02）

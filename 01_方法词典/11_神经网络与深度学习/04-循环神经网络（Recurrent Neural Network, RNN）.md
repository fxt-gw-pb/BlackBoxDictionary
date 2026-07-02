---
title: 循环神经网络
english_name: Recurrent Neural Network, RNN
slug: recurrent-neural-network-rnn
aliases: [RNN, recurrent neural network, "循环神经网络（Recurrent Neural Network, RNN）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: intermediate
question_type: 序列依赖建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 循环神经网络（Recurrent Neural Network, RNN）

## 1. 方法概览

### 1.1 定义

循环神经网络是一类面向序列数据的神经网络。它在每个时间步共享同一组参数，并把前一时刻的隐藏状态传递到当前时刻，从而建模顺序依赖关系。

### 1.2 它主要解决什么问题

- 研究问题：当样本有明确时间顺序或位置顺序时，如何利用历史上下文预测当前或未来输出。
- 适用任务：时间序列预测、序列分类、文本建模、事件序列建模。
- 常见医学场景：生命体征序列分析、电子病历时间窗口风险预测、医疗文本序列标注。

### 1.3 直觉理解

RNN 像一个一边读序列一边做笔记的模型。它在每个时间点读取新信息，同时把“上一步记住的东西”带到下一步，用来辅助当前判断。

## 2. 数学形式

### 2.1 核心公式

对时间步 $t$，基础 RNN 可写为：

$$
h_t = \phi(W_x x_t + W_h h_{t-1} + b_h)
$$

$$
\hat y_t = g(W_y h_t + b_y)
$$

其中 $h_t$ 为隐藏状态，$\phi$ 常取 `tanh` 或 `relu`，$g$ 依任务而定。训练通常通过时间反向传播（BPTT）完成。

### 2.2 参数或统计量含义

- $x_t$：第 $t$ 个时间步输入。
- $h_t$：第 $t$ 个时间步隐藏状态。
- $W_x, W_h, W_y$：输入到隐藏、隐藏到隐藏、隐藏到输出的权重。
- 序列长度：决定模型可见上下文窗口。

### 2.3 关键假设

- 序列顺序有信息价值。
- 相邻或远距离时间点之间存在可学习依赖。
- 训练数据足以支持时序模式学习。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单变量或多变量序列、token 序列、事件序列。
- 因变量类型：当前时刻标签、未来值或整段序列标签。
- 数据结构：三维张量，通常为样本数乘以时间步乘以特征数。
- 是否适合高维数据：可以，但长序列和高维输入会显著增加训练难度。
- 是否适合缺失较多数据：需先插补或引入缺失掩码。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：适合规则重复测量；不规则间隔需额外时间信息。

### 3.2 示例表格

以 ICU 每小时生命体征序列分类为例：

| PatientID | Hour | HR | RR | SpO2 | ShockIn6h |
| --- | --- | --- | --- | --- | --- |
| P001 | 1 | 94 | 20 | 97 | 0 |
| P001 | 2 | 101 | 22 | 96 | 0 |
| P001 | 3 | 112 | 26 | 94 | 1 |
| P002 | 1 | 78 | 16 | 99 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：按顺序排列的序列张量和标签。
- 关键变量：序列长度、隐藏单元数、层数、学习率。
- 需要预处理的内容：时间排序、标准化、窗口构造、mask 处理。

#### 产出

- 模型对象/统计结果：训练好的 RNN 参数和验证指标。
- 参数估计：共享循环权重。
- 预测结果：分类标签、概率、序列表示或连续预测值。
- 不确定性指标：验证集误差、分层性能、外部时间段测试表现。

## 4. 适用场景

- 适合：需要利用先后顺序和上下文信息的任务。
- 不适合：序列很长且依赖跨度大、并行效率要求高的场景，此时常被 Transformer 取代。
- 使用前需要特别检查的点：时间泄露、长序列梯度消失、样本独立划分。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch
from torch import nn

class SimpleRNNClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=32, n_classes=2):
        super().__init__()
        self.rnn = nn.RNN(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_simple_rnn(units = 32, input_shape = c(seq_len, n_features)) |>
  layer_dense(units = 2, activation = "softmax")
```

## 6. 结果如何解释

- 核心结果看什么：验证集性能、不同时间窗口下的稳定性、是否过拟合。
- 每个主要参数如何解释：隐藏单元数和层数反映模型容量，序列长度影响可见历史范围。
- 临床或医学意义如何表达：更适合用于动态风险预测，但需要明确预测时间窗和干预时点。
- 常见误读：模型能利用顺序信息，不代表它一定真正学到长期依赖。

## 7. 推荐可视化

- 预测概率随时间变化曲线。
- 训练/验证损失曲线。
- 不同序列长度下的性能比较图。

## 8. 优势、局限与常见坑

### 优势

- 能直接处理顺序输入。
- 参数在各时间步共享，结构自然。
- 是理解门控时序网络的基础。

### 局限

- 容易出现梯度消失或爆炸。
- 对长距离依赖支持有限。
- 并行效率不如 Transformer。

### 常见坑

- 随机拆分序列，导致相邻样本泄露。
- 用普通 RNN 处理很长序列却不做梯度裁剪。
- 只在单个时间段测试，忽略时移漂移。

## 9. 与相近方法的区别

- 和 [[长短期记忆网络（Long Short-Term Memory, LSTM）]] 的区别：LSTM 在基础 RNN 上引入门控和记忆单元，更适合长依赖。
- 和 [[门控循环单元（Gated Recurrent Unit, GRU）]] 的区别：GRU 结构更紧凑，训练常更高效。
- 和 [[Transformer（Transformer）]] 的区别：RNN 按时间步递归处理，Transformer 用注意力并行建模全局依赖。

## 10. 医学研究中的典型应用

- ICU 时序监测数据的短期风险预测。
- 门诊随访序列的结局建模。
- 医疗文本或事件流序列分析。

## 11. 相关方法

- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[门控循环单元（Gated Recurrent Unit, GRU）]]
- [[Transformer（Transformer）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]

## 12. 参考资料

- Elman JL. Finding structure in time. *Cognitive Science*. 1990;14(2):179-211.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- PyTorch Developers. `torch.nn.RNN`. [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) （访问日期：2026-07-02）

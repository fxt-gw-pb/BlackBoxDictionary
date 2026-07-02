---
title: 长短期记忆网络
english_name: Long Short-Term Memory, LSTM
slug: long-short-term-memory-lstm
aliases: [LSTM, long short-term memory, "长短期记忆网络（Long Short-Term Memory, LSTM）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 长程序列依赖建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 长短期记忆网络（Long Short-Term Memory, LSTM）

## 1. 方法概览

### 1.1 定义

LSTM 是循环神经网络的重要变体，通过遗忘门、输入门和输出门控制信息流动，专门用于缓解普通 RNN 难以学习长程依赖的问题。

### 1.2 它主要解决什么问题

- 研究问题：当序列历史较长、关键信息可能跨越多个时间步时，如何保留有用记忆并抑制无关信息。
- 适用任务：时间序列预测、序列分类、文本建模、事件序列分析。
- 常见医学场景：ICU 连续监测数据预测、药物处方序列分析、长期随访事件风险建模。

### 1.3 直觉理解

LSTM 像一个带有“记忆抽屉”和“开关阀门”的序列模型。每到一个新时间点，它都会决定什么该忘掉、什么该写进去、什么该输出，因此比普通 RNN 更能处理远距离依赖。

## 2. 数学形式

### 2.1 核心公式

对时间步 $t$，LSTM 的门控更新为：

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i),\qquad
\tilde c_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o),\qquad
h_t = o_t \odot \tanh(c_t)
$$

### 2.2 参数或统计量含义

- $f_t$：遗忘门，控制旧记忆保留比例。
- $i_t$：输入门，控制新信息写入比例。
- $o_t$：输出门，控制当前隐藏状态暴露程度。
- $c_t$：细胞状态，是长期记忆通道。
- $h_t$：隐藏状态，是当前时刻对外输出的表征。

### 2.3 关键假设

- 远距离依赖对预测有价值。
- 序列顺序和时间结构不能被打乱。
- 训练样本足够支持门控参数学习。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单变量或多变量序列、token 序列、事件流。
- 因变量类型：未来值、序列标签、下一个事件或整个输出序列。
- 数据结构：三维序列张量。
- 是否适合高维数据：可以，但训练和正则化要求更高。
- 是否适合缺失较多数据：需先做插补、缺失掩码或专门时序缺失建模。
- 是否适合删失数据：不直接建模删失机制。
- 是否适合重复测量数据：适合有顺序的重复测量，尤其是规则采样。

### 3.2 示例表格

以连续血糖预测为例：

| PatientID | Time | Glucose | InsulinDose | CarbIntake | NextGlucoseRisk |
| --- | --- | --- | --- | --- | --- |
| P001 | 08:00 | 7.1 | 4 | 35 | normal |
| P001 | 09:00 | 8.4 | 0 | 0 | high |
| P001 | 10:00 | 9.2 | 2 | 10 | high |
| P002 | 08:00 | 5.8 | 3 | 25 | normal |

### 3.3 输入与产出

#### 输入

- 输入数据：固定窗口或整段序列张量。
- 关键变量：窗口长度、隐藏单元数、层数、dropout、学习率。
- 需要预处理的内容：排序、标准化、时间切分、窗口构造、避免未来信息泄露。

#### 产出

- 模型对象/统计结果：训练好的 LSTM 权重、损失曲线、验证性能。
- 参数估计：门控层和输出层权重。
- 预测结果：类别概率、连续值或下一步序列输出。
- 不确定性指标：验证误差、时间外测试、bootstrap 或 ensemble 近似区间。

## 4. 适用场景

- 适合：存在长短期依赖、普通 RNN 难以稳定建模的序列任务。
- 不适合：序列很短、样本太少、或更需要并行效率和全局依赖时。
- 使用前需要特别检查的点：窗口设计、过拟合、训练不稳定、外部时移漂移。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_lstm(units = 64, input_shape = c(seq_len, n_features)) |>
  layer_dense(units = 2, activation = "softmax")
```

## 6. 结果如何解释

- 核心结果看什么：验证误差、不同预测窗口的稳定性、长时间外测试表现。
- 每个主要参数如何解释：隐藏单元数和层数控制容量，窗口长度控制模型可见历史。
- 临床或医学意义如何表达：适合动态风险更新，但必须明确预测时间点和可干预时间。
- 常见误读：LSTM 的“记忆”是统计意义上的状态表示，不等同于可直接解释的临床记忆机制。

## 7. 推荐可视化

- 真实值与预测值时间曲线。
- 训练/验证损失曲线。
- 分时间窗或分患者亚组的误差图。

## 8. 优势、局限与常见坑

### 优势

- 比基础 RNN 更擅长捕捉长程依赖。
- 对复杂多变量序列有较强表达能力。
- 已广泛用于时间序列和文本任务。

### 局限

- 训练和调参成本较高。
- 可解释性有限。
- 仍然存在并行效率不高的问题。

### 常见坑

- 用随机拆分代替按时间或按患者拆分。
- 序列太短却堆叠过深网络。
- 只在内部验证集上报告效果。

## 9. 与相近方法的区别

- 和 [[循环神经网络（Recurrent Neural Network, RNN）]] 的区别：LSTM 用门控机制缓解梯度消失，适合更长依赖。
- 和 [[门控循环单元（Gated Recurrent Unit, GRU）]] 的区别：GRU 结构更紧凑，参数更少，训练常更快。
- 和 [[Transformer（Transformer）]] 的区别：Transformer 通过注意力并行建模全局依赖，LSTM 仍以递归顺序处理序列。

## 10. 医学研究中的典型应用

- 连续血糖、心率和血压等生理信号预测。
- 长随访电子病历事件序列建模。
- ICU 短期恶化风险动态更新。

## 11. 相关方法

- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[门控循环单元（Gated Recurrent Unit, GRU）]]
- [[Transformer（Transformer）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]

## 12. 参考资料

- Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Computation*. 1997;9(8):1735-1780.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- PyTorch Developers. `torch.nn.LSTM`. [https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) （访问日期：2026-07-02）

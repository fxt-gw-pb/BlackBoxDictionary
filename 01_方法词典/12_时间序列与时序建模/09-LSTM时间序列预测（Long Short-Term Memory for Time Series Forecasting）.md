---
title: LSTM时间序列预测
english_name: Long Short-Term Memory for Time Series Forecasting
slug: long-short-term-memory-for-time-series-forecasting
aliases: [LSTM time series forecasting, long short-term memory for time series forecasting, "LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）"]
category: 时间序列与时序建模
subcategory: 时序深度学习
tags: [医学统计, 数据科学, 时间序列, 深度学习, 神经网络, 预测]
status: 已建
difficulty: advanced
question_type: 非线性序列依赖建模与预测
data_type: [时间序列数据, 多变量时间序列数据, 序列数据]
outcome_type: [连续型, 时间序列, 多分类]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）

## 1. 方法概览

### 1.1 定义

LSTM 是循环神经网络的一种改进结构，通过门控机制保存、遗忘和输出序列信息。本条目聚焦 [[长短期记忆网络（Long Short-Term Memory, LSTM）]] 在时间序列预测中的应用型变体，强调窗口构造、时间切分和预测评估。

### 1.2 它主要解决什么问题

- 研究问题：序列中存在非线性、长短期依赖或多变量复杂模式时，如何进行预测。
- 适用任务：多步预测、生命体征序列建模、传感器数据预测、序列分类或异常检测。
- 常见医学场景：ICU 生命体征预测、连续血糖预测、可穿戴设备序列分析、EHR 时间窗口风险预测。

### 1.3 直觉理解

LSTM 像一个带记忆的预测器。它每读入一个时间点，就决定哪些历史信息要保留、哪些要忘掉、哪些要输出给当前预测，因此比普通线性模型更能表示复杂序列模式。

## 2. 数学形式

### 2.1 核心公式

给定输入 $x_t$、上一隐藏状态 $h_{t-1}$ 和细胞状态 $c_{t-1}$，LSTM 的门控更新为：

$$
f_t=\sigma(W_f x_t+U_f h_{t-1}+b_f)
$$

$$
i_t=\sigma(W_i x_t+U_i h_{t-1}+b_i)
$$

$$
\tilde c_t=\tanh(W_c x_t+U_c h_{t-1}+b_c)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t
$$

$$
o_t=\sigma(W_o x_t+U_o h_{t-1}+b_o),\quad
h_t=o_t\odot \tanh(c_t)
$$

预测输出可写为：

$$
\hat y_{t+h}=g(W_y h_t+b_y)
$$

### 2.2 参数或统计量含义

- $f_t$：遗忘门，控制保留多少旧记忆。
- $i_t$：输入门，控制写入多少新信息。
- $o_t$：输出门，控制输出多少当前记忆。
- $c_t$：细胞状态，表示长期记忆。
- $h_t$：隐藏状态，表示当前输出表征。
- lookback window：输入历史窗口长度。

### 2.3 关键假设

- 训练数据量足以支持神经网络拟合。
- 未来数据分布与训练期相近。
- 输入窗口包含预测所需的主要历史信息。
- 时间顺序严格保留，预处理不能泄露未来信息。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单变量或多变量历史序列窗口。
- 因变量类型：未来连续值、类别标签或序列。
- 数据结构：三维张量，通常为样本数乘以时间步乘以特征数。
- 是否适合高维数据：可处理多特征序列，但需要足够样本和正则化。
- 是否适合缺失较多数据：需插补、掩码或专门缺失建模。
- 是否适合删失数据：不直接处理删失结局，需结合生存模型扩展。
- 是否适合重复测量数据：适合规则采样序列；不规则间隔需时间间隔特征或专门模型。

### 3.2 示例表格

以小时级生命体征为例：

| PatientID | Time | HeartRate | MAP | SpO2 | Lactate |
| --- | --- | --- | --- | --- | --- |
| P001 | 2026-01-01 00:00 | 88 | 74 | 97 | 1.8 |
| P001 | 2026-01-01 01:00 | 91 | 72 | 96 | 1.9 |
| P001 | 2026-01-01 02:00 | 95 | 70 | 95 | 2.1 |
| P001 | 2026-01-01 03:00 | 90 | 73 | 96 | 2.0 |

### 3.3 输入与产出

#### 输入

- 输入数据：按时间排序的序列窗口张量。
- 关键变量：窗口长度、预测步长、隐藏单元数、层数、学习率、batch size。
- 需要预处理的内容：时间切分、标准化、缺失处理、窗口构造、避免未来信息泄露。

#### 产出

- 模型对象/统计结果：训练好的神经网络、损失曲线、预测值。
- 参数估计：网络权重和偏置。
- 预测结果：未来值、类别概率或多步预测序列。
- 不确定性指标：验证集误差、bootstrap/ensemble 不确定性、预测区间近似。

## 4. 适用场景

- 适合：数据量较大、非线性明显、多变量序列复杂、传统线性模型拟合不足的场景。
- 不适合：样本很少、解释性要求极高、序列规律简单且线性模型已足够的场景。
- 使用前需要特别检查的点：时间切分、标准化 fit 范围、过拟合、外部验证、临床可解释性。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])

# X_train: [n_samples, lookback, n_features]
# y_train: [n_samples, 1]
model = LSTMForecaster(n_features=X_train.shape[2])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    pred = model(torch.tensor(X_train, dtype=torch.float32))
    loss = loss_fn(pred, torch.tensor(y_train, dtype=torch.float32))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_lstm(units = 32, input_shape = c(lookback, n_features)) |>
  layer_dense(units = 1)

model |>
  compile(optimizer = "adam", loss = "mse")

history <- model |>
  fit(X_train, y_train, epochs = 50, batch_size = 32, validation_split = 0.2)
```

## 6. 结果如何解释

- 核心结果看什么：验证集/测试集误差、训练损失曲线、预测曲线、外部验证表现。
- 每个主要参数如何解释：窗口长度决定模型能看到多长历史，隐藏单元数决定模型容量。
- 临床或医学意义如何表达：可强调其预测性能，但需配合解释工具、误差分析和临床安全边界。
- 常见误读：LSTM 预测好不代表学到了因果机制，也不代表在新医院或新设备上同样有效。

## 7. 推荐可视化

- 实际值与预测值时间曲线。
- 训练和验证损失曲线。
- 不同预测步长的误差曲线。
- 分层误差分析图，如按患者亚组或病区分层。

## 8. 优势、局限与常见坑

### 优势

- 能表达复杂非线性和长短期依赖。
- 适合多变量序列和高频监测数据。
- 可扩展到多步预测、序列分类和异常检测。

### 局限

- 数据需求大，训练成本高。
- 可解释性弱。
- 对预处理、窗口构造和验证设计非常敏感。

### 常见坑

- 标准化时使用全数据造成信息泄露。
- 随机拆分窗口导致同一患者相邻时间泄露到测试集。
- 只报告训练误差，不做时间外或中心外验证。

## 9. 与相近方法的区别

- 和 [[长短期记忆网络（Long Short-Term Memory, LSTM）]] 的关系：本条目是通用 LSTM 架构在时间序列预测场景下的具体应用版本。
- 和 [[向量自回归模型（Vector Autoregression, VAR）]] 的区别：VAR 是线性多变量模型，LSTM 是非线性神经网络。
- 和 [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]] 的区别：XGBoost 依赖显式滞后特征表格化，LSTM 直接读取序列窗口。
- 和 [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]] 的区别：MLP 不内置时间递归结构，LSTM 专门为序列依赖设计。

## 10. 医学研究中的典型应用

- ICU 生命体征和实验室指标的短期预测。
- 连续血糖、心率或可穿戴设备信号预测。
- EHR 时间窗口中的风险预警模型。

## 11. 相关方法

- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[向量自回归模型（Vector Autoregression, VAR）]]
- [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]]
- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]

## 12. 参考资料

- Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Computation*. 1997;9(8):1735-1780.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- PyTorch Developers. `torch.nn.LSTM`. [https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) （访问日期：2026-07-02）

---
title: 移动平均模型
english_name: Moving Average Model, MA
slug: moving-average-model-ma
aliases: [MA, moving average model, 移动平均模型, "移动平均模型（Moving Average Model, MA）"]
category: 时间序列与时序建模
subcategory: 经典单变量时间序列
tags: [医学统计, 数据科学, 时间序列, 预测, 平滑]
status: 已建
difficulty: basic
question_type: 短期冲击建模与时间序列平滑
data_type: [时间序列数据, 纵向数据]
outcome_type: [连续型, 时间序列]
python_packages: [statsmodels, pandas]
r_packages: [stats, forecast]
---

# 移动平均模型（Moving Average Model, MA）

## 1. 方法概览

### 1.1 定义

移动平均模型在时间序列建模中通常指 MA($q$)：当前值由当前随机冲击和过去 $q$ 个随机冲击线性组合而成。实践中“移动平均”也常指简单移动平均平滑，用于去噪和观察趋势。

### 1.2 它主要解决什么问题

- 研究问题：短期随机冲击是否会在随后几个时间点继续影响序列。
- 适用任务：误差相关建模、时间序列平滑、ARMA/ARIMA 的基础组件。
- 常见医学场景：短期异常就诊峰值平滑，监测指标噪声过滤，感染病例报告延迟造成的短期波动建模。

### 1.3 直觉理解

AR 模型看过去的观测值，MA 模型看过去未被解释的“意外冲击”。如果一次突发事件会让接下来几期都偏高或偏低，MA 项就可以描述这种短期冲击传播。

## 2. 数学形式

### 2.1 核心公式

MA($q$) 模型写作：

$$
y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
$$

其中：

$$
\varepsilon_t\sim WN(0,\sigma^2)
$$

简单移动平均平滑常写为：

$$
\operatorname{SMA}_t=\frac{1}{m}\sum_{j=0}^{m-1}y_{t-j}
$$

### 2.2 参数或统计量含义

- $q$：移动平均阶数，表示过去多少期冲击进入模型。
- $\theta_j$：第 $j$ 个滞后冲击的影响系数。
- $\varepsilon_t$：创新项或白噪声误差。
- $m$：简单移动平均窗口长度。
- ACF：MA 模型阶数识别中常用的自相关图。

### 2.3 关键假设

- 序列或残差过程近似平稳。
- 创新项为白噪声。
- 冲击影响有限，超过 $q$ 期后消失。
- 简单移动平均平滑会引入滞后，不应用于因果解释。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：过去误差项或滚动窗口内历史观测。
- 因变量类型：连续型时间序列。
- 数据结构：等间隔单变量时间序列。
- 是否适合高维数据：不适合直接处理多变量高维。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级时间序列；多受试者需专门纵向模型。

### 3.2 示例表格

以每周流感样病例数为例：

| Week | ILICases |
| --- | --- |
| 2026-W01 | 320 |
| 2026-W02 | 355 |
| 2026-W03 | 410 |
| 2026-W04 | 390 |

### 3.3 输入与产出

#### 输入

- 输入数据：按时间排序的单变量序列。
- 关键变量：MA 阶数 $q$，或平滑窗口长度 $m$。
- 需要预处理的内容：缺失处理、频率对齐、平稳性检查、异常峰值标注。

#### 产出

- 模型对象/统计结果：MA 系数、残差、拟合值、预测值。
- 参数估计：$\theta_1,\dots,\theta_q$ 和误差方差。
- 预测结果：短期预测和预测区间。
- 不确定性指标：标准误、预测区间、残差诊断。

## 4. 适用场景

- 适合：短期冲击明显、残差自相关有限、需要平滑噪声观察趋势的序列。
- 不适合：长期趋势或季节性强而未处理、冲击持续时间很长、非线性变化明显的序列。
- 使用前需要特别检查的点：MA 阶数、残差白噪声、平滑窗口是否造成过度滞后。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`
- `pandas`

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("weekly_ili_cases.csv")
y = pd.Series(df["ILICases"].values, index=pd.PeriodIndex(df["Week"], freq="W"))

smooth = y.rolling(window=4, min_periods=1).mean()
fit = ARIMA(y.astype(float), order=(0, 0, 2)).fit()
forecast = fit.forecast(steps=4)

print(smooth.tail())
print(forecast)
```

### 5.2 R

常用包：

- `stats`
- `forecast`

```r
library(forecast)

y <- ts(df$ILICases, frequency = 52)
smooth <- stats::filter(y, rep(1 / 4, 4), sides = 1)
fit <- Arima(y, order = c(0, 0, 2), include.mean = TRUE)

forecast(fit, h = 4)
```

## 6. 结果如何解释

- 核心结果看什么：MA 系数、残差 ACF、预测误差、平滑曲线与原始曲线差异。
- 每个主要参数如何解释：$\theta_1$ 表示上一期未解释冲击对当前值的影响。
- 临床或医学意义如何表达：适合描述短期报告噪声、突发冲击或监测指标平滑趋势。
- 常见误读：简单移动平均曲线不是预测模型的充分证据，也不能消除真实延迟偏倚。

## 7. 推荐可视化

- 原始序列与移动平均平滑曲线。
- ACF 图。
- 残差时间图。
- 预测值与实际值对比图。

## 8. 优势、局限与常见坑

### 优势

- 能描述短期冲击影响。
- 平滑版本直观易懂。
- 是 ARMA/ARIMA 的重要组成部分。

### 局限

- 纯 MA 模型解释长期结构能力有限。
- 简单移动平均窗口选择主观。
- 平滑会削弱峰值并引入滞后。

### 常见坑

- 混淆“MA 模型”和“简单移动平均平滑”。
- 使用居中移动平均时不小心引入未来信息。
- 窗口过大导致疫情或就诊峰值被抹平。

## 9. 与相近方法的区别

- 和 [[自回归模型（Autoregressive Model, AR）]] 的区别：AR 使用过去观测值，MA 使用过去创新误差。
- 和 [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]] 的区别：ARMA 同时建模历史观测和历史误差。
- 和 [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]] 的区别：Holt-Winters 专门处理水平、趋势和季节性平滑。

## 10. 医学研究中的典型应用

- 传染病病例数的短期报告噪声平滑。
- 医院服务量监测中的短期异常波动识别。
- 生理监测序列中短期测量噪声过滤。

## 11. 相关方法

- [[自回归模型（Autoregressive Model, AR）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]]

## 12. 参考资料

- Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley; 2015.
- Brockwell PJ, Davis RA. *Introduction to Time Series and Forecasting*. 3rd ed. Springer; 2016.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）

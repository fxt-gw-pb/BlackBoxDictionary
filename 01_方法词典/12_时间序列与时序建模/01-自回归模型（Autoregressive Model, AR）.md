---
title: 自回归模型
english_name: Autoregressive Model, AR
slug: autoregressive-model-ar
aliases: [AR, autoregressive model, 自回归, "自回归模型（Autoregressive Model, AR）"]
category: 时间序列与时序建模
subcategory: 经典单变量时间序列
tags: [医学统计, 数据科学, 时间序列, 预测, 自相关]
status: 已建
difficulty: basic
question_type: 单变量平稳时间序列预测
data_type: [时间序列数据, 纵向数据]
outcome_type: [连续型, 时间序列]
python_packages: [statsmodels]
r_packages: [stats, forecast]
---

# 自回归模型（Autoregressive Model, AR）

## 1. 方法概览

### 1.1 定义

自回归模型是一类用序列自身过去值预测当前值的时间序列模型。AR($p$) 表示当前观测由前 $p$ 个滞后值的线性组合和随机误差构成。

### 1.2 它主要解决什么问题

- 研究问题：某个连续指标是否能由自己的历史水平预测。
- 适用任务：短期预测、自相关建模、时间序列基线模型、ARMA/ARIMA 的基础组件。
- 常见医学场景：每日门诊量、连续生命体征、医院床位占用率、空气污染暴露指标的短期预测。

### 1.3 直觉理解

如果今天的指标和昨天、前天高度相关，就可以用过去值来预测今天。AR 模型把这种“惯性”写成一个线性回归，只是自变量来自同一条时间序列的滞后值。

## 2. 数学形式

### 2.1 核心公式

AR($p$) 模型写作：

$$
y_t=c+\phi_1 y_{t-1}+\phi_2 y_{t-2}+\cdots+\phi_p y_{t-p}+\varepsilon_t
$$

其中白噪声误差满足：

$$
\varepsilon_t\sim WN(0,\sigma^2)
$$

使用滞后算子 $B$ 可写为：

$$
\phi(B)y_t=c+\varepsilon_t,\quad
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p
$$

### 2.2 参数或统计量含义

- $p$：自回归阶数，表示使用多少个历史滞后值。
- $\phi_j$：第 $j$ 阶滞后值对当前值的线性影响。
- $c$：截距或均值项。
- $\varepsilon_t$：不可预测的新信息或创新项。
- ACF/PACF：帮助识别自相关结构和阶数。

### 2.3 关键假设

- 序列近似平稳，均值、方差和自协方差不随时间系统变化。
- 残差近似白噪声。
- 主要依赖关系可由线性滞后项表达。
- 采样间隔固定，时间顺序不能打乱。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：目标变量的滞后值。
- 因变量类型：连续型时间序列。
- 数据结构：等间隔单变量时间序列。
- 是否适合高维数据：不适合直接处理高维多变量。
- 是否适合缺失较多数据：需先补齐或使用支持缺失的状态空间估计。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合单个总体序列；多受试者重复测量需面板或混合模型。

### 3.2 示例表格

以每日急诊量为例：

| Date | EDVisits |
| --- | --- |
| 2026-01-01 | 212 |
| 2026-01-02 | 225 |
| 2026-01-03 | 231 |
| 2026-01-04 | 220 |

### 3.3 输入与产出

#### 输入

- 输入数据：按时间排序的单变量序列。
- 关键变量：滞后阶数 $p$、是否包含截距、训练预测窗口。
- 需要预处理的内容：缺失处理、频率对齐、平稳性检查、异常值检查。

#### 产出

- 模型对象/统计结果：AR 系数、残差、拟合值、预测值。
- 参数估计：$\phi_1,\dots,\phi_p$、截距和误差方差。
- 预测结果：未来若干期点预测和预测区间。
- 不确定性指标：标准误、置信区间、预测区间、残差诊断。

## 4. 适用场景

- 适合：单变量、近似平稳、短期惯性强的时间序列。
- 不适合：强趋势、强季节性、突变频繁或明显非线性的序列，除非先做变换或扩展。
- 使用前需要特别检查的点：平稳性、残差自相关、滞后阶数、预测是否只使用过去信息。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

df = pd.read_csv("daily_ed_visits.csv", parse_dates=["Date"])
y = df.set_index("Date")["EDVisits"].asfreq("D")
y = y.interpolate()

fit = AutoReg(y, lags=7, old_names=False).fit()
forecast = fit.predict(start=len(y), end=len(y) + 6)

print(fit.summary())
print(forecast)
```

### 5.2 R

常用包：

- `stats`
- `forecast`

```r
library(forecast)

y <- ts(df$EDVisits, frequency = 7)
fit <- arima(y, order = c(7, 0, 0), include.mean = TRUE)

pred <- forecast(fit, h = 7)
summary(fit)
pred
```

## 6. 结果如何解释

- 核心结果看什么：滞后系数、残差是否白噪声、短期预测误差。
- 每个主要参数如何解释：$\phi_1$ 表示上一期值对当前值的线性贡献，其他滞后系数同理。
- 临床或医学意义如何表达：可用于说明指标存在时间惯性，并给出短期资源需求预测。
- 常见误读：显著滞后系数不代表过去值“因果导致”当前值。

## 7. 推荐可视化

- 原始序列和拟合/预测曲线。
- ACF 与 PACF 图。
- 残差时间图和残差 ACF。
- 滚动预测误差曲线。

## 8. 优势、局限与常见坑

### 优势

- 结构简单，解释清楚。
- 适合作为时间序列预测基线。
- 与 ACF/PACF 诊断结合紧密。

### 局限

- 要求近似平稳。
- 难以处理季节性、趋势和非线性。
- 对结构突变敏感。

### 常见坑

- 随机划分训练测试集，破坏时间顺序。
- 未检查平稳性和残差自相关。
- 用未来信息构造滞后特征。

## 9. 与相近方法的区别

- 和 [[移动平均模型（Moving Average Model, MA）]] 的区别：AR 使用过去观测值，MA 使用过去误差项。
- 和 [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]] 的区别：ARMA 同时包含 AR 和 MA 部分。
- 和 [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]] 的区别：ARIMA 加入差分以处理非平稳序列。

## 10. 医学研究中的典型应用

- 每日急诊量、门诊量或住院人数短期预测。
- 连续监测指标的短期趋势建模。
- 环境暴露或传染病病例数的平稳阶段预测。

## 11. 相关方法

- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[向量自回归模型（Vector Autoregression, VAR）]]

## 12. 参考资料

- Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley; 2015.
- Hamilton JD. *Time Series Analysis*. Princeton University Press; 1994.
- statsmodels Developers. `statsmodels.tsa.ar_model.AutoReg`. [https://www.statsmodels.org/](https://www.statsmodels.org/) （访问日期：2026-07-02）

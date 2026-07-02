---
title: Holt-Winters指数平滑
english_name: Holt-Winters Exponential Smoothing
slug: holt-winters-exponential-smoothing
aliases: [Holt-Winters, triple exponential smoothing, 指数平滑, "Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）"]
category: 时间序列与时序建模
subcategory: 指数平滑
tags: [医学统计, 数据科学, 时间序列, 预测, 季节性, 指数平滑]
status: 已建
difficulty: basic
question_type: 带趋势和季节性的单变量序列预测
data_type: [时间序列数据]
outcome_type: [连续型, 时间序列]
python_packages: [statsmodels]
r_packages: [forecast, stats]
---

# Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）

## 1. 方法概览

### 1.1 定义

Holt-Winters 指数平滑是一种用于带趋势和季节性的单变量时间序列预测方法。它递推更新水平、趋势和季节性三个成分，并用它们生成未来预测。

### 1.2 它主要解决什么问题

- 研究问题：具有稳定季节周期和趋势的序列如何快速预测。
- 适用任务：季节性业务指标预测、平滑和短中期预测、运营需求预测。
- 常见医学场景：月度门诊量、季节性呼吸道疾病就诊量、药品消耗、检验项目需求预测。

### 1.3 直觉理解

Holt-Winters 把每个时间点的信息拆成当前水平、增长趋势和季节位置。新观测到来时，它让最近数据权重更高，逐步更新这三个成分。

## 2. 数学形式

### 2.1 核心公式

加法 Holt-Winters 模型：

$$
\ell_t=\alpha(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1})
$$

$$
b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1}
$$

$$
s_t=\gamma(y_t-\ell_t)+(1-\gamma)s_{t-m}
$$

预测公式：

$$
\hat y_{t+h|t}=\ell_t+hb_t+s_{t+h-m(k+1)}
$$

其中 $m$ 为季节周期长度。

### 2.2 参数或统计量含义

- $\ell_t$：水平项。
- $b_t$：趋势项。
- $s_t$：季节性项。
- $\alpha$：水平平滑参数。
- $\beta$：趋势平滑参数。
- $\gamma$：季节性平滑参数。
- $m$：季节周期，如月度数据中的 12。

### 2.3 关键假设

- 季节周期相对稳定。
- 趋势和季节性可由指数平滑递推表达。
- 加法模型适合季节波动幅度大致恒定的序列；乘法模型适合季节波动随水平放大的序列。
- 序列频率规则且周期长度明确。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：时间索引和历史序列。
- 因变量类型：连续型单变量时间序列。
- 数据结构：等间隔、具有季节周期的序列。
- 是否适合高维数据：不适合直接处理高维。
- 是否适合缺失较多数据：需先补齐或插补。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级序列，不适合直接处理个体重复测量。

### 3.2 示例表格

以月度呼吸科门诊量为例：

| Month | RespiratoryVisits |
| --- | --- |
| 2025-01 | 3220 |
| 2025-02 | 2980 |
| 2025-03 | 2760 |
| 2025-04 | 2410 |

### 3.3 输入与产出

#### 输入

- 输入数据：单变量等间隔季节性序列。
- 关键变量：季节周期、加法/乘法季节性、平滑参数。
- 需要预处理的内容：缺失处理、异常值检查、频率设定、训练测试时间切分。

#### 产出

- 模型对象/统计结果：水平、趋势、季节性成分、拟合值。
- 参数估计：$\alpha,\beta,\gamma$ 和初始状态。
- 预测结果：未来值预测和预测区间。
- 不确定性指标：预测区间、残差诊断、滚动预测误差。

## 4. 适用场景

- 适合：单变量、趋势和季节性稳定、需要快速可解释预测的序列。
- 不适合：多个季节周期、强外部事件影响、季节性频繁改变或序列很短的场景。
- 使用前需要特别检查的点：季节周期是否足够多、加法/乘法选择、残差是否有自相关。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("monthly_respiratory_visits.csv", parse_dates=["Month"])
y = df.set_index("Month")["RespiratoryVisits"].asfreq("MS")

fit = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=12
).fit()

forecast = fit.forecast(12)
print(forecast)
```

### 5.2 R

常用包：

- `forecast`

```r
library(forecast)

y <- ts(df$RespiratoryVisits, frequency = 12)
fit <- hw(y, seasonal = "additive", h = 12)

summary(fit)
fit
```

## 6. 结果如何解释

- 核心结果看什么：水平、趋势、季节性成分、预测误差和残差结构。
- 每个主要参数如何解释：$\alpha$ 越大，模型越快响应最新水平变化。
- 临床或医学意义如何表达：适合解释某类服务需求的季节高峰和长期变化趋势。
- 常见误读：季节性稳定是模型假设，不能自动适应重大政策、疫情或诊疗流程变化。

## 7. 推荐可视化

- 原始序列和预测曲线。
- 分解出的水平、趋势和季节性成分。
- 季节图或月份箱线图。
- 残差时间图和残差 ACF。

## 8. 优势、局限与常见坑

### 优势

- 简单、快速、解释性好。
- 对稳定趋势和季节性序列表现稳健。
- 不需要复杂特征工程。

### 局限

- 难以处理复杂外部变量。
- 对多季节周期和结构突变表达有限。
- 序列太短时季节性估计不稳定。

### 常见坑

- 月度序列不足两个完整季节周期就拟合季节模型。
- 加法/乘法季节性选错。
- 不做时间切分验证。

## 9. 与相近方法的区别

- 和 [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]] 的区别：ARIMA 强调差分和自相关，Holt-Winters 强调水平、趋势和季节性递推平滑。
- 和 [[Prophet时间序列模型（Prophet Forecasting Model）]] 的区别：Prophet 支持变化点和节假日效应，Holt-Winters 更轻量。
- 和 [[移动平均模型（Moving Average Model, MA）]] 的区别：简单移动平均只平滑窗口内值，Holt-Winters 同时递推趋势和季节性。

## 10. 医学研究中的典型应用

- 呼吸科、儿科等季节性明显科室的门诊量预测。
- 药品、耗材、床位需求的月度预测。
- 传染病季节性高峰的运营准备。

## 11. 相关方法

- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[Prophet时间序列模型（Prophet Forecasting Model）]]
- [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]]

## 12. 参考资料

- Holt CC. Forecasting seasonals and trends by exponentially weighted moving averages. *International Journal of Forecasting*. 2004;20(1):5-10.
- Winters PR. Forecasting sales by exponentially weighted moving averages. *Management Science*. 1960;6(3):324-342.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）

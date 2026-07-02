---
title: 自回归滑动平均模型
english_name: Autoregressive Moving Average Model, ARMA
slug: autoregressive-moving-average-model-arma
aliases: [ARMA, autoregressive moving average, "自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）"]
category: 时间序列与时序建模
subcategory: 经典单变量时间序列
tags: [医学统计, 数据科学, 时间序列, 预测, 平稳序列]
status: 已建
difficulty: intermediate
question_type: 平稳序列自相关与冲击结构建模
data_type: [时间序列数据, 纵向数据]
outcome_type: [连续型, 时间序列]
python_packages: [statsmodels]
r_packages: [forecast, stats]
---

# 自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）

## 1. 方法概览

### 1.1 定义

ARMA 模型把自回归 AR 部分和移动平均 MA 部分组合起来，用于建模平稳单变量时间序列。ARMA($p,q$) 同时利用过去观测值和过去误差项预测当前值。

### 1.2 它主要解决什么问题

- 研究问题：平稳序列中既有历史惯性，又有短期冲击传播时，如何统一建模。
- 适用任务：平稳时间序列预测、残差相关建模、ARIMA 的无差分特例。
- 常见医学场景：平稳阶段的床位占用率、标准化后的空气污染指标、去趋势后的病例数或监测指标。

### 1.3 直觉理解

ARMA 同时问两件事：过去观测水平会不会影响现在，过去没解释掉的冲击会不会继续影响现在。它比单独 AR 或 MA 更灵活，但仍要求序列近似平稳。

## 2. 数学形式

### 2.1 核心公式

ARMA($p,q$) 模型为：

$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}
+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}
$$

滞后算子形式为：

$$
\phi(B)y_t=c+\theta(B)\varepsilon_t
$$

其中：

$$
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\quad
\theta(B)=1+\theta_1B+\cdots+\theta_qB^q
$$

### 2.2 参数或统计量含义

- $p$：AR 阶数。
- $q$：MA 阶数。
- $\phi_i$：自回归系数。
- $\theta_j$：移动平均系数。
- AIC/BIC：常用于比较不同阶数组合。
- ACF/PACF：用于初步判断 AR 与 MA 结构。

### 2.3 关键假设

- 序列平稳。
- 残差为白噪声。
- 模型满足稳定性和可逆性条件。
- 线性滞后结构足以描述主要时间依赖。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：目标变量滞后值和误差滞后项。
- 因变量类型：连续型平稳时间序列。
- 数据结构：等间隔单变量时间序列。
- 是否适合高维数据：不适合直接处理多变量高维。
- 是否适合缺失较多数据：需先处理缺失或使用支持缺失的估计方法。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级序列，不适合直接处理个体重复测量。

### 3.2 示例表格

以去趋势后的 PM2.5 日均值为例：

| Date | PM25_detrended |
| --- | --- |
| 2026-01-01 | -3.2 |
| 2026-01-02 | 1.8 |
| 2026-01-03 | 4.1 |
| 2026-01-04 | -0.7 |

### 3.3 输入与产出

#### 输入

- 输入数据：平稳或已去趋势的单变量序列。
- 关键变量：AR 阶数 $p$、MA 阶数 $q$、是否包含常数项。
- 需要预处理的内容：平稳性检查、缺失处理、异常值检查、训练测试时间切分。

#### 产出

- 模型对象/统计结果：AR/MA 系数、残差、信息准则、预测值。
- 参数估计：$\phi_i$、$\theta_j$、误差方差。
- 预测结果：未来若干期点预测和预测区间。
- 不确定性指标：参数标准误、预测区间、残差白噪声检验。

## 4. 适用场景

- 适合：平稳单变量序列，ACF/PACF 显示有限滞后依赖。
- 不适合：明显趋势或季节性未处理、结构突变、强非线性、外部协变量很重要的场景。
- 使用前需要特别检查的点：平稳性、阶数选择、残差诊断、预测窗口是否合理。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("daily_pm25_detrended.csv", parse_dates=["Date"])
y = df.set_index("Date")["PM25_detrended"].asfreq("D").interpolate()

fit = ARIMA(y, order=(2, 0, 1)).fit()
forecast = fit.get_forecast(steps=7)

print(fit.summary())
print(forecast.summary_frame())
```

### 5.2 R

常用包：

- `forecast`

```r
library(forecast)

y <- ts(df$PM25_detrended, frequency = 7)
fit <- Arima(y, order = c(2, 0, 1), include.mean = TRUE)

forecast(fit, h = 7)
checkresiduals(fit)
```

## 6. 结果如何解释

- 核心结果看什么：AR/MA 系数、AIC/BIC、残差是否白噪声、预测误差。
- 每个主要参数如何解释：AR 系数反映历史观测影响，MA 系数反映历史冲击影响。
- 临床或医学意义如何表达：适合描述去趋势后指标的短期自相关结构，并用于短期预测。
- 常见误读：ARMA 对平稳序列建模，不应直接套在有明显趋势的原始序列上。

## 7. 推荐可视化

- 原始/去趋势序列图。
- ACF 与 PACF 图。
- 拟合值和预测值对比图。
- 残差 ACF、QQ 图和残差时间图。

## 8. 优势、局限与常见坑

### 优势

- 统一表达 AR 和 MA 依赖。
- 对平稳线性序列预测效果稳健。
- 模型诊断体系成熟。

### 局限

- 不能直接处理非平稳趋势。
- 阶数选择需要诊断和比较。
- 对结构突变和非线性关系表达有限。

### 常见坑

- 忽略平稳性要求。
- 只靠 AIC 选模型而不看残差。
- 在模型选择中反复看测试集造成乐观偏差。

## 9. 与相近方法的区别

- 和 [[自回归模型（Autoregressive Model, AR）]] 的区别：ARMA 额外加入 MA 误差项。
- 和 [[移动平均模型（Moving Average Model, MA）]] 的区别：ARMA 额外加入过去观测值。
- 和 [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]] 的区别：ARIMA 通过差分处理非平稳序列，ARMA 适合已平稳序列。

## 10. 医学研究中的典型应用

- 去趋势后的空气污染或温度暴露预测。
- 医院运行指标的平稳阶段短期预测。
- 传染病监测残差相关结构建模。

## 11. 相关方法

- [[自回归模型（Autoregressive Model, AR）]]
- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[向量自回归模型（Vector Autoregression, VAR）]]

## 12. 参考资料

- Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley; 2015.
- Brockwell PJ, Davis RA. *Time Series: Theory and Methods*. 2nd ed. Springer; 1991.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）

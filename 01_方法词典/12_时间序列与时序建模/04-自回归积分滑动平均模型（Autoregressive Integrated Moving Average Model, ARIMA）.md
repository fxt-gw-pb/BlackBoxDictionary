---
title: 自回归积分滑动平均模型
english_name: Autoregressive Integrated Moving Average Model, ARIMA
slug: autoregressive-integrated-moving-average-model-arima
aliases: [ARIMA, autoregressive integrated moving average, "自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）"]
category: 时间序列与时序建模
subcategory: 经典单变量时间序列
tags: [医学统计, 数据科学, 时间序列, 预测, 差分, 非平稳序列]
status: 已建
difficulty: intermediate
question_type: 非平稳单变量时间序列预测
data_type: [时间序列数据, 纵向数据]
outcome_type: [连续型, 时间序列]
python_packages: [statsmodels, pmdarima]
r_packages: [forecast, stats]
---

# 自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）

## 1. 方法概览

### 1.1 定义

ARIMA 是用于单变量时间序列预测的经典模型。它通过差分把非平稳序列转为平稳序列，再用 ARMA 模型刻画差分后序列的自相关结构。

### 1.2 它主要解决什么问题

- 研究问题：带趋势或单位根的单变量序列如何建模和预测。
- 适用任务：短中期预测、趋势序列建模、监测指标预测、ARMA 的非平稳扩展。
- 常见医学场景：每日就诊量、住院人数、传染病病例数、药品消耗量、环境暴露指标预测。

### 1.3 直觉理解

如果原始序列一直上升或下降，直接建 ARMA 往往不合适。ARIMA 先看“变化量”而不是原始水平，等变化量稳定后再建模，最后把预测变化量累加回原始尺度。

## 2. 数学形式

### 2.1 核心公式

ARIMA($p,d,q$) 对序列做 $d$ 次差分：

$$
w_t=(1-B)^d y_t
$$

再对 $w_t$ 建 ARMA($p,q$)：

$$
\phi(B)w_t=c+\theta(B)\varepsilon_t
$$

整体写作：

$$
\phi(B)(1-B)^d y_t=c+\theta(B)\varepsilon_t
$$

其中：

$$
\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\quad
\theta(B)=1+\theta_1B+\cdots+\theta_qB^q
$$

### 2.2 参数或统计量含义

- $p$：自回归阶数。
- $d$：差分阶数。
- $q$：移动平均阶数。
- $\phi_i$：AR 系数。
- $\theta_j$：MA 系数。
- AIC/BIC：模型阶数比较指标。
- ADF/KPSS：平稳性检验常用工具。

### 2.3 关键假设

- 差分后序列近似平稳。
- 残差近似白噪声。
- 主要时间依赖可由线性 ARMA 结构捕捉。
- 结构关系在预测期内大致稳定。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：目标变量自身滞后值、差分值和误差滞后项。
- 因变量类型：连续型单变量时间序列。
- 数据结构：等间隔时间序列。
- 是否适合高维数据：不适合直接处理多变量高维。
- 是否适合缺失较多数据：需处理缺失；状态空间实现可处理部分缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级序列，不适合直接处理个体重复测量。

### 3.2 示例表格

以每日住院人数为例：

| Date | Inpatients |
| --- | --- |
| 2026-01-01 | 842 |
| 2026-01-02 | 851 |
| 2026-01-03 | 865 |
| 2026-01-04 | 859 |

### 3.3 输入与产出

#### 输入

- 输入数据：按固定频率排列的单变量序列。
- 关键变量：$p,d,q$ 阶数、是否包含 drift 或截距、预测步长。
- 需要预处理的内容：缺失处理、异常值标记、变换、平稳性检查、训练测试时间切分。

#### 产出

- 模型对象/统计结果：AR/MA 系数、差分阶数、残差、信息准则。
- 参数估计：$\phi_i$、$\theta_j$、误差方差和截距/漂移项。
- 预测结果：未来值预测、预测区间。
- 不确定性指标：参数标准误、预测区间、残差诊断。

## 4. 适用场景

- 适合：单变量、趋势性明显但差分后平稳、需要可解释短中期预测的序列。
- 不适合：多重季节性、强外部驱动变量、复杂非线性、频繁结构突变的序列。
- 使用前需要特别检查的点：是否过度差分、残差是否白噪声、预测区间是否合理、时间切分是否正确。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("daily_inpatients.csv", parse_dates=["Date"])
y = df.set_index("Date")["Inpatients"].asfreq("D").interpolate()

train = y.iloc[:-14]
test = y.iloc[-14:]

fit = ARIMA(train, order=(2, 1, 2)).fit()
forecast = fit.get_forecast(steps=14).summary_frame()

print(fit.summary())
print(forecast.head())
```

### 5.2 R

常用包：

- `forecast`

```r
library(forecast)

y <- ts(df$Inpatients, frequency = 7)
fit <- auto.arima(y, seasonal = FALSE)

pred <- forecast(fit, h = 14)
summary(fit)
checkresiduals(fit)
pred
```

## 6. 结果如何解释

- 核心结果看什么：差分阶数、AR/MA 系数、AIC/BIC、残差诊断、滚动预测误差。
- 每个主要参数如何解释：$d=1$ 表示对原序列做一阶差分后建模，即建模相邻时间点变化量。
- 临床或医学意义如何表达：可用于短期资源预测，如未来 1-2 周住院量或门诊量变化。
- 常见误读：ARIMA 预测延续历史结构，不会自动知道政策变化、节假日或突发事件。

## 7. 推荐可视化

- 原始序列与差分后序列图。
- ACF/PACF 图。
- 预测曲线和预测区间。
- 残差诊断图和滚动预测误差图。

## 8. 优势、局限与常见坑

### 优势

- 经典、透明、诊断体系成熟。
- 适合单变量短期预测。
- 能处理一类差分后平稳的非平稳序列。

### 局限

- 难以处理复杂外部因素和非线性。
- 多季节性和长周期结构需扩展模型。
- 预测远期时不确定性快速增大。

### 常见坑

- 过度差分导致序列过度噪声化。
- 只依赖 auto.arima，不看残差诊断。
- 随机划分训练测试集。

## 9. 与相近方法的区别

- 和 [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]] 的区别：ARMA 要求序列平稳，ARIMA 通过差分处理非平稳。
- 和 [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]] 的区别：Holt-Winters 直接分解水平、趋势和季节性，ARIMA 通过差分和自相关结构建模。
- 和 [[Prophet时间序列模型（Prophet Forecasting Model）]] 的区别：Prophet 更强调趋势变化点、季节性和节假日效应，ARIMA 更强调自相关结构。

## 10. 医学研究中的典型应用

- 医院每日住院量、急诊量、药品需求量预测。
- 传染病病例数或监测指标短期预测。
- 环境暴露时间序列趋势建模。

## 11. 相关方法

- [[自回归模型（Autoregressive Model, AR）]]
- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]]
- [[Prophet时间序列模型（Prophet Forecasting Model）]]

## 12. 参考资料

- Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis: Forecasting and Control*. 5th ed. Wiley; 2015.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）
- statsmodels Developers. `statsmodels.tsa.arima.model.ARIMA`. [https://www.statsmodels.org/](https://www.statsmodels.org/) （访问日期：2026-07-02）

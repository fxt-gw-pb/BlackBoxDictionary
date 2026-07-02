---
title: 向量自回归模型
english_name: Vector Autoregression, VAR
slug: vector-autoregression-var
aliases: [VAR, vector autoregression, 向量自回归, "向量自回归模型（Vector Autoregression, VAR）"]
category: 时间序列与时序建模
subcategory: 多变量时间序列
tags: [医学统计, 数据科学, 时间序列, 多变量, 预测, 脉冲响应]
status: 已建
difficulty: intermediate
question_type: 多变量时间序列联动建模
data_type: [多变量时间序列数据, 纵向数据]
outcome_type: [连续型, 多变量时间序列]
python_packages: [statsmodels]
r_packages: [vars]
---

# 向量自回归模型（Vector Autoregression, VAR）

## 1. 方法概览

### 1.1 定义

向量自回归模型是多变量时间序列模型，用每个变量自身和其他变量的历史值共同预测当前多个变量。VAR($p$) 可看作多变量版 AR 模型。

### 1.2 它主要解决什么问题

- 研究问题：多个时间序列之间是否存在滞后联动关系。
- 适用任务：多指标联合预测、动态关联分析、脉冲响应分析、预测误差方差分解。
- 常见医学场景：感染病例、就诊量与气象污染指标联动；多项生命体征动态预测；医院运营指标之间的相互影响分析。

### 1.3 直觉理解

VAR 不只问“这个指标过去能否预测自己”，还问“其他指标的过去能否帮助预测它”。例如气温、污染和病例数可能互相滞后影响，VAR 用一个系统同时建模这些关系。

## 2. 数学形式

### 2.1 核心公式

设 $y_t\in\mathbb{R}^K$ 为 $K$ 维时间序列向量。VAR($p$) 写作：

$$
y_t=c+A_1y_{t-1}+A_2y_{t-2}+\cdots+A_py_{t-p}+u_t
$$

其中：

$$
u_t\sim WN(0,\Sigma_u)
$$

$h$ 步预测递推为：

$$
\hat y_{T+h}=c+\sum_{i=1}^{p}A_i \hat y_{T+h-i}
$$

### 2.2 参数或统计量含义

- $K$：变量数量。
- $p$：滞后阶数。
- $A_i$：第 $i$ 阶滞后系数矩阵。
- $\Sigma_u$：创新项协方差矩阵。
- IRF：脉冲响应函数，描述一个变量受到冲击后系统的动态响应。
- FEVD：预测误差方差分解。

### 2.3 关键假设

- 多个序列联合平稳，或已通过差分/变换处理。
- 滞后阶数足以捕捉主要动态关系。
- 残差近似白噪声。
- 变量数量和滞后阶数不能相对样本量过大。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：多个变量的滞后值。
- 因变量类型：多个连续型时间序列。
- 数据结构：等间隔多变量时间序列，每行一个时间点。
- 是否适合高维数据：变量多时参数急剧增加，需降维或正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级多序列；个体级重复测量需面板 VAR 或混合模型。

### 3.2 示例表格

以公共卫生监测为例：

| Date | ILICases | EDVisits | PM25 | Temperature |
| --- | --- | --- | --- | --- |
| 2026-01-01 | 320 | 212 | 48 | 2.1 |
| 2026-01-02 | 355 | 225 | 52 | 1.8 |
| 2026-01-03 | 410 | 231 | 60 | 0.4 |
| 2026-01-04 | 390 | 220 | 55 | 3.0 |

### 3.3 输入与产出

#### 输入

- 输入数据：多个同步时间序列。
- 关键变量：滞后阶数、趋势/截距设定、预测步长。
- 需要预处理的内容：缺失处理、频率对齐、平稳性检查、变量变换、滞后阶数选择。

#### 产出

- 模型对象/统计结果：系数矩阵、残差协方差、预测值、IRF、FEVD。
- 参数估计：$A_1,\dots,A_p$、截距和残差协方差。
- 预测结果：多个变量的联合预测和区间。
- 不确定性指标：参数标准误、预测区间、IRF 置信区间。

## 4. 适用场景

- 适合：多变量指标同步记录、变量之间可能存在滞后影响、目标是系统动态解释和预测的场景。
- 不适合：变量太多样本太少、非平稳且无协整处理、强非线性或结构突变明显的场景。
- 使用前需要特别检查的点：平稳性、滞后阶数、残差相关、变量顺序对正交化 IRF 的影响。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.api import VAR

df = pd.read_csv("public_health_monitoring.csv", parse_dates=["Date"])
y = df.set_index("Date")[["ILICases", "EDVisits", "PM25", "Temperature"]].asfreq("D")
y = y.interpolate()

model = VAR(y)
fit = model.fit(maxlags=7, ic="aic")
forecast = fit.forecast(y.values[-fit.k_ar:], steps=7)

print(fit.summary())
print(pd.DataFrame(forecast, columns=y.columns))
```

### 5.2 R

常用包：

- `vars`

```r
library(vars)

y <- ts(df[, c("ILICases", "EDVisits", "PM25", "Temperature")], frequency = 7)
lag_select <- VARselect(y, lag.max = 7, type = "const")
fit <- VAR(y, p = lag_select$selection["AIC(n)"], type = "const")

predict(fit, n.ahead = 7)
serial.test(fit)
```

## 6. 结果如何解释

- 核心结果看什么：滞后系数矩阵、联合预测、IRF、FEVD、残差诊断。
- 每个主要参数如何解释：$A_i[j,k]$ 表示第 $k$ 个变量滞后 $i$ 期对第 $j$ 个变量当前值的线性贡献。
- 临床或医学意义如何表达：可说明多个监测指标之间的滞后联动，但需谨慎区分预测关联和因果效应。
- 常见误读：VAR 系数不是自动的因果效应，IRF 也依赖识别设定。

## 7. 推荐可视化

- 多变量时间序列折线图。
- 预测曲线和预测区间。
- 脉冲响应函数图。
- 残差相关热图和残差 ACF。

## 8. 优势、局限与常见坑

### 优势

- 能同时建模多个序列的动态关系。
- 不需要事先指定严格因果方向。
- 可做 IRF 和 FEVD 分析。

### 局限

- 参数数量随变量数和滞后阶数快速增加。
- 对平稳性敏感。
- 解释因果关系需要额外识别假设。

### 常见坑

- 变量太多但样本太短。
- 忽略单位根或协整问题。
- 过度解读滞后系数的因果含义。

## 9. 与相近方法的区别

- 和 [[自回归模型（Autoregressive Model, AR）]] 的区别：AR 是单变量，VAR 是多变量系统。
- 和 [[向量自回归滑动平均模型（Vector Autoregressive Moving Average Model, VARMA）]] 的区别：VARMA 还包含多变量误差移动平均项。
- 和 [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]] 的区别：VAR 是线性系统模型，XGBoost 依赖特征工程和树模型捕捉非线性。

## 10. 医学研究中的典型应用

- 传染病病例、就诊量、气象和污染指标的动态联动分析。
- ICU 多项生命体征短期联合预测。
- 医院运营指标之间的滞后影响探索。

## 11. 相关方法

- [[自回归模型（Autoregressive Model, AR）]]
- [[向量自回归滑动平均模型（Vector Autoregressive Moving Average Model, VARMA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]]

## 12. 参考资料

- Lutkepohl H. *New Introduction to Multiple Time Series Analysis*. Springer; 2005.
- Sims CA. Macroeconomics and reality. *Econometrica*. 1980;48(1):1-48.
- statsmodels Developers. Vector Autoregressions documentation. [https://www.statsmodels.org/](https://www.statsmodels.org/) （访问日期：2026-07-02）

---
title: Prophet时间序列模型
english_name: Prophet Forecasting Model
slug: prophet-forecasting-model
aliases: [Prophet, Facebook Prophet, prophet forecasting, "Prophet时间序列模型（Prophet Forecasting Model）"]
category: 时间序列与时序建模
subcategory: 可加性预测模型
tags: [医学统计, 数据科学, 时间序列, 预测, 趋势, 季节性]
status: 已建
difficulty: intermediate
question_type: 带趋势变化点和季节性的业务时间序列预测
data_type: [时间序列数据]
outcome_type: [连续型, 时间序列]
python_packages: [prophet]
r_packages: [prophet]
---

# Prophet时间序列模型（Prophet Forecasting Model）

## 1. 方法概览

### 1.1 一句话本质

Prophet 把序列拆成分段趋势、周期季节、节假日与误差，让每种可解释结构单独建模后相加。

### 1.2 定义

Prophet 是一种面向业务时间序列预测的可加性模型。它把序列分解为趋势、季节性、节假日效应和误差项，并通过变化点捕捉趋势斜率的改变。

### 1.3 它主要解决什么问题

- 研究问题：带趋势、季节性、节假日或政策节点的时间序列如何快速建模和预测。
- 适用任务：运营指标预测、日/周/月频序列预测、可解释趋势分解、缺失和异常较多序列的稳健预测。
- 常见医学场景：门诊量、急诊量、检验量、药品消耗、节假日影响下的医院服务需求预测。

### 1.4 直觉理解

Prophet 把时间序列看成几种可解释成分的叠加：长期趋势、周期性波动、节假日影响和随机噪声。它的优势是把许多业务中常见的时间结构显式拆开。

## 2. 核心思想与原理

业务序列常有趋势变点、多个季节周期与缺测。Prophet 的关键洞察是用可加性分解和稀疏变点调整代替单一全局趋势，使人工已知日历效应也能显式进入模型。

## 3. 数学形式

### 3.1 核心公式

Prophet 的核心可加形式为：

$$
y(t)=g(t)+s(t)+h(t)+\varepsilon_t
$$

其中 $g(t)$ 是趋势，$s(t)$ 是季节性，$h(t)$ 是节假日效应。分段线性趋势可写作：

$$
g(t)=(k+a(t)^\top\delta)t+(m+a(t)^\top\gamma)
$$

季节性通常用 Fourier 级数表示：

$$
s(t)=\sum_{n=1}^{N}\left[
a_n\cos\left(\frac{2\pi nt}{P}\right)+
b_n\sin\left(\frac{2\pi nt}{P}\right)
\right]
$$

### 3.3 参数或统计量含义

- $g(t)$：趋势项，可为线性或 logistic 增长。
- $s(t)$：季节性项，如周季节性、年季节性。
- $h(t)$：节假日或特殊日期效应。
- `changepoint_prior_scale`：趋势变化点灵活度。
- `seasonality_prior_scale`：季节性灵活度。
- `interval_width`：预测区间宽度。

### 3.4 关键假设

- 序列可由趋势、季节性、节假日和噪声的可加结构近似。
- 历史趋势变化规律对未来有参考价值。
- 特殊事件可通过节假日或回归变量显式标注。
- 预测区间主要来自模型设定下的不确定性，不覆盖所有外部突发风险。

## 4. 手把手算例

某日基础趋势为 100，周末季节效应 $-12$，节假日效应 $+20$：
$$
\hat y=100-12+20=108
$$
若下一日趋势升至 102、非周末且无节假日，则预测为 102。这个算例点明 Prophet 的预测是各结构组件相加，而非“黑箱自动猜测”。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：日期时间、节假日、额外回归变量。
- 因变量类型：连续型时间序列。
- 数据结构：至少包含 `ds` 时间列和 `y` 数值列。
- 是否适合高维数据：不适合直接处理高维特征；额外回归变量需谨慎选择。
- 是否适合缺失较多数据：能处理不规则缺失时间点，但长期缺口仍需谨慎。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合单条业务序列；多个个体序列需分层或批量建模。

### 5.2 示例表格

以每日门诊量为例：

| ds | y | holiday |
| --- | --- | --- |
| 2026-01-01 | 1820 | NewYear |
| 2026-01-02 | 2460 | none |
| 2026-01-03 | 2310 | none |
| 2026-01-04 | 1980 | none |

### 5.3 输入与产出

#### 输入

- 输入数据：`ds` 和 `y` 两列，外加可选节假日、容量上限、回归变量。
- 关键变量：趋势类型、变化点灵活度、季节性设置、节假日表。
- 需要预处理的内容：异常值标记、缺失检查、频率确认、特殊事件编码。

#### 产出

- 模型对象/统计结果：趋势、季节性、节假日效应、预测值。
- 参数估计：趋势变化点、季节性 Fourier 系数、节假日效应。
- 预测结果：`yhat`、`yhat_lower`、`yhat_upper`。
- 不确定性指标：预测区间、交叉验证误差、成分不确定性。

## 6. 适用场景

- 适合：业务运营序列、明显周/年季节性、节假日效应明显、需要快速可解释预测的场景。
- 不适合：强自相关残差、复杂多变量动态系统、短序列、突发结构变化无法标注的场景。
- 使用前需要特别检查的点：变化点灵活度、节假日编码、交叉验证误差、残差是否仍有自相关。

## 7. 实现

### 7.1 Python

常用包：

- `prophet`

```python
import pandas as pd
from prophet import Prophet

df = pd.read_csv("daily_outpatient_visits.csv", parse_dates=["ds"])
holidays = pd.DataFrame({
    "holiday": ["NewYear"],
    "ds": pd.to_datetime(["2026-01-01"]),
    "lower_window": [0],
    "upper_window": [1]
})

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    holidays=holidays,
    changepoint_prior_scale=0.05
)
model.fit(df[["ds", "y"]])

future = model.make_future_dataframe(periods=14, freq="D")
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
```

### 7.2 R

常用包：

- `prophet`

```r
library(prophet)

df$ds <- as.Date(df$ds)
fit <- prophet(
  df[, c("ds", "y")],
  weekly.seasonality = TRUE,
  yearly.seasonality = TRUE,
  changepoint.prior.scale = 0.05
)

future <- make_future_dataframe(fit, periods = 14)
forecast <- predict(fit, future)
tail(forecast[, c("ds", "yhat", "yhat_lower", "yhat_upper")])
```

## 8. 结果如何解释

- 核心结果看什么：预测曲线、预测区间、趋势变化点、周/年季节性和节假日效应。
- 每个主要参数如何解释：`changepoint_prior_scale` 越大，趋势越灵活，也越可能过拟合。
- 临床或医学意义如何表达：可用于解释服务量在工作日、季节和节假日前后的系统性变化。
- 常见误读：Prophet 自动拟合趋势，不代表它能预测未出现过的政策或疫情冲击。

## 9. 假设诊断与稳健性

做滚动起点验证，检查变点先验、季节周期、节假日窗口、异常值和预测区间覆盖；疫情或政策造成的永久结构变化不能只靠默认参数。

## 10. 推荐可视化

- 历史值与预测值曲线。
- 趋势、周季节性、年季节性和节假日成分图。
- 交叉验证误差随预测 horizon 变化图。
- 残差时间图。

## 11. 优势、局限与常见坑

### 优势

- 对趋势、季节性和节假日建模直观。
- 对缺失日期和异常点相对稳健。
- 上手快，适合运营预测。

### 局限

- 对强自相关和复杂动态系统表达有限。
- 短序列难以稳定估计季节性。
- 需要认真配置节假日和变化点。

### 常见坑

- 不做时间序列交叉验证。
- 变化点设置过灵活导致过拟合。
- 把默认节假日直接用于医疗场景而不结合本地排班和政策。

## 12. 与相近方法的区别

- 和 [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]] 的区别：ARIMA 强调自相关结构，Prophet 强调趋势、季节性和节假日可加分解。
- 和 [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]] 的区别：Holt-Winters 用递推平滑水平、趋势和季节性，Prophet 允许变化点和节假日效应。
- 和 [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]] 的区别：XGBoost 依赖人工构造滞后和日历特征，Prophet 内置时间成分结构。

## 13. 医学研究中的典型应用

- 医院门诊量、急诊量和检查量预测。
- 节假日和工作日模式下的医疗资源需求预测。
- 药品、耗材和床位需求的短中期运营预测。

## 14. 关键术语

- **变点（Changepoint）**：趋势斜率允许改变的时点。
- **可加性分解**：趋势、季节和节假日组件相加。
- **Fourier 项**：用正弦余弦表示平滑周期。
- **滚动起点验证**：按时间推进训练终点的回测。

## 15. 相关方法

- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]]
- [[XGBoost时间序列预测（XGBoost for Time Series Forecasting）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]

## 16. 参考资料

- Taylor SJ, Letham B. Forecasting at scale. *The American Statistician*. 2018;72(1):37-45.
- Prophet Developers. Prophet documentation. [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/) （访问日期：2026-07-02）
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）

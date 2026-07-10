---
title: 向量自回归滑动平均模型
english_name: Vector Autoregressive Moving Average Model, VARMA
slug: vector-autoregressive-moving-average-model-varma
aliases: [VARMA, vector autoregressive moving average, "向量自回归滑动平均模型（Vector Autoregressive Moving Average Model, VARMA）"]
category: 时间序列与时序建模
subcategory: 多变量时间序列
tags: [医学统计, 数据科学, 时间序列, 多变量, 状态空间, 预测]
status: 已建
difficulty: advanced
question_type: 多变量序列自相关与冲击结构建模
data_type: [多变量时间序列数据, 纵向数据]
outcome_type: [连续型, 多变量时间序列]
python_packages: [statsmodels]
r_packages: [MTS]
---

# 向量自回归滑动平均模型（Vector Autoregressive Moving Average Model, VARMA）

## 1. 方法概览

### 1.1 一句话本质

VARMA 同时用多变量过去水平与过去联合冲击预测现在，是 VAR 与多变量 MA 的合体。

### 1.2 定义

VARMA 是多变量时间序列模型，同时包含向量自回归 VAR 部分和向量移动平均 VMA 部分。它既用多个变量的历史值，也用多个变量的过去创新项解释当前向量序列。

### 1.3 它主要解决什么问题

- 研究问题：多个时间序列之间既有滞后联动，又有短期共同冲击传播时，如何联合建模。
- 适用任务：多变量预测、系统动态建模、状态空间估计、冲击响应分析。
- 常见医学场景：多项生命体征联动预测，医院运营指标联合预测，污染-气象-病例系统动态建模。

### 1.4 直觉理解

VARMA 比 VAR 更灵活：VAR 看“过去观测”，VARMA 还看“过去没有解释掉的冲击”。如果多个指标受到共同突发冲击，而且冲击会在系统中短期延续，VARMA 可以表达这种结构。

## 2. 核心思想与原理

VAR 只让冲击通过状态递推传播；VARMA 还允许冲击在有限期内直接留下回声。关键困难是 MA 冲击不可观测、参数可辨识性弱，因此模型虽简约，估计与选阶比 VAR 更难。

## 3. 数学形式

### 3.1 核心公式

设 $y_t\in\mathbb{R}^K$，VARMA($p,q$) 写作：

$$
y_t=c+\sum_{i=1}^{p}A_i y_{t-i}
+u_t+\sum_{j=1}^{q}M_j u_{t-j}
$$

其中：

$$
u_t\sim WN(0,\Sigma_u)
$$

滞后算子形式为：

$$
A(B)y_t=c+M(B)u_t
$$

其中：

$$
A(B)=I-A_1B-\cdots-A_pB^p,\quad
M(B)=I+M_1B+\cdots+M_qB^q
$$

### 3.3 参数或统计量含义

- $p$：向量自回归阶数。
- $q$：向量移动平均阶数。
- $A_i$：第 $i$ 阶滞后观测系数矩阵。
- $M_j$：第 $j$ 阶滞后创新系数矩阵。
- $\Sigma_u$：创新协方差矩阵。
- 状态空间表示：常用于估计、滤波和预测。

### 3.4 关键假设

- 多变量序列平稳或已处理为平稳。
- 模型满足稳定性和可逆性。
- 参数可识别，阶数选择合理。
- 样本量足以支持较多参数估计。

## 4. 手把手算例

设 $\mathbf y_t=0.5\mathbf y_{t-1}+\boldsymbol\varepsilon_t+0.3\boldsymbol\varepsilon_{t-1}$，$\mathbf y_0=\mathbf0$。若 $\varepsilon_1=(2,1)$、$\varepsilon_2=(-1,0)$，则
$$
\mathbf y_1=(2,1),\qquad
\mathbf y_2=0.5(2,1)+(-1,0)+0.3(2,1)=(0.6,0.8)
$$
第二期同时包含水平惯性 $(1,0.5)$、新冲击 $(-1,0)$ 与旧冲击回声 $(0.6,0.3)$。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：多个变量的滞后值和滞后创新项。
- 因变量类型：多个连续时间序列。
- 数据结构：等间隔多变量时间序列。
- 是否适合高维数据：变量多时参数爆炸，通常不适合高维。
- 是否适合缺失较多数据：状态空间方法可处理部分缺失，但需谨慎。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：适合总体级多序列；个体级需面板扩展。

### 5.2 示例表格

以 ICU 小时级监测为例：

| Time | HeartRate | MAP | SpO2 | RespiratoryRate |
| --- | --- | --- | --- | --- |
| 2026-01-01 00:00 | 88 | 74 | 97 | 18 |
| 2026-01-01 01:00 | 91 | 72 | 96 | 19 |
| 2026-01-01 02:00 | 95 | 70 | 95 | 21 |
| 2026-01-01 03:00 | 90 | 73 | 96 | 18 |

### 5.3 输入与产出

#### 输入

- 输入数据：多个同步时间序列。
- 关键变量：$p,q$ 阶数、趋势项、状态空间估计设置。
- 需要预处理的内容：频率对齐、缺失处理、平稳性检查、变量标准化、阶数选择。

#### 产出

- 模型对象/统计结果：AR/MA 系数矩阵、残差协方差、预测值、状态估计。
- 参数估计：$A_i$、$M_j$、$\Sigma_u$。
- 预测结果：多变量未来路径和预测区间。
- 不确定性指标：标准误、预测区间、残差诊断、IRF 不确定性。

## 6. 适用场景

- 适合：多变量序列有复杂短期冲击传播，VAR 残差仍有结构，样本量足够的场景。
- 不适合：样本短、变量多、模型识别困难、只需要简单预测基线的场景。
- 使用前需要特别检查的点：可识别性、收敛、残差白噪声、阶数选择、估计稳定性。

## 7. 实现

### 7.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX

df = pd.read_csv("icu_hourly_vitals.csv", parse_dates=["Time"])
y = df.set_index("Time")[["HeartRate", "MAP", "SpO2", "RespiratoryRate"]].asfreq("H")
y = y.interpolate()

fit = VARMAX(y, order=(1, 1), trend="c").fit(disp=False, maxiter=200)
forecast = fit.get_forecast(steps=6).summary_frame()

print(fit.summary())
print(forecast.head())
```

### 7.2 R

常用包：

- `MTS`

```r
library(MTS)

y <- as.matrix(df[, c("HeartRate", "MAP", "SpO2", "RespiratoryRate")])
fit <- VARMA(y, p = 1, q = 1)

fit
```

## 8. 结果如何解释

- 核心结果看什么：AR 矩阵、MA 矩阵、残差相关、预测表现、模型是否收敛。
- 每个主要参数如何解释：AR 矩阵描述过去观测联动，MA 矩阵描述过去创新冲击的传播。
- 临床或医学意义如何表达：可用于描述多项指标动态系统，但解释需谨慎，特别是冲击响应依赖模型设定。
- 常见误读：VARMA 更复杂不一定预测更好，尤其在样本量不足时。

## 9. 假设诊断与稳健性

检查联合平稳性、可逆性、残差相关、参数近边界和滚动预测；与更简单 VAR 比较，避免不可辨识的过度参数化。

## 10. 推荐可视化

- 多变量原始序列图。
- 多变量预测路径和区间。
- 残差相关热图。
- 脉冲响应和预测误差方差分解图。

## 11. 优势、局限与常见坑

### 优势

- 比 VAR 更灵活，能刻画短期创新传播。
- 可用状态空间框架处理估计和预测。
- 适合多变量动态系统建模。

### 局限

- 参数多，估计困难。
- 识别和收敛问题常见。
- 对样本量和平稳性要求高。

### 常见坑

- 在短序列上拟合高阶 VARMA。
- 忽略收敛警告。
- 不做残差诊断就解释系数。

## 12. 与相近方法的区别

- 和 [[向量自回归模型（Vector Autoregression, VAR）]] 的区别：VARMA 增加向量移动平均项，能表达创新冲击的滞后传播。
- 和 [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]] 的区别：ARMA 是单变量，VARMA 是多变量系统。
- 和 [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]] 的区别：VARMA 是线性统计模型，LSTM 是非线性神经网络模型。

## 13. 医学研究中的典型应用

- ICU 多生命体征联合预测。
- 医院多运营指标动态建模。
- 环境暴露、气象因素和病例数的多变量系统分析。

## 14. 关键术语

- **矩阵滞后多项式**：用矩阵统一表示多变量 AR/MA 动态。
- **可辨识性**：不同参数是否对应不同观测分布。
- **创新协方差**：同一时点各变量新冲击的相关结构。
- **可逆性**：可由观测稳定恢复创新。

## 15. 相关方法

- [[向量自回归模型（Vector Autoregression, VAR）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]
- [[Prophet时间序列模型（Prophet Forecasting Model）]]

## 16. 参考资料

- Lutkepohl H. *New Introduction to Multiple Time Series Analysis*. Springer; 2005.
- Tsay RS. *Multivariate Time Series Analysis: With R and Financial Applications*. Wiley; 2014.
- statsmodels Developers. `statsmodels.tsa.statespace.varmax.VARMAX`. [https://www.statsmodels.org/](https://www.statsmodels.org/) （访问日期：2026-07-02）

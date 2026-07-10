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
### 1.1 一句话本质
AR 用序列自身过去的值预测现在，把“惯性会延续但逐渐衰减”写成线性递推。
### 1.2 定义
AR($p$) 以最近 $p$ 个滞后值解释当前值，并把其余不可预测部分视为白噪声。
### 1.3 它主要解决什么问题
- 平稳连续序列的短期预测与自相关刻画。
- 医学场景：按日就诊量、稳定期生命体征、实验室质量控制。
### 1.4 直觉与类比
今天偏高会把明天也“拉高”；系数决定记忆保留多少，反复递推后影响通常衰减。

## 2. 核心思想与原理
### 2.1 根本困难
相邻观测并不独立，普通均值模型浪费了时间依赖。
### 2.2 关键洞察
将当前值投影到过去值张成的空间；剩余残差若近似白噪声，说明可预测结构已被提取。
### 2.3 与朴素做法对比
延用最后值相当于固定 $\phi=1$，通常非平稳；AR 从数据估计记忆强度。

## 3. 数学形式
### 3.1 核心公式
$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\varepsilon_t,\qquad
\varepsilon_t\sim WN(0,\sigma^2)
$$
AR(1) 平稳时 $\mu=c/(1-\phi)$、$\operatorname{Var}(y_t)=\sigma^2/(1-\phi^2)$。
### 3.2 推导脉络
最小化一步预测平方误差可估计系数；平稳性要求特征根在单位圆外，AR(1) 即 $|\phi|<1$。
### 3.3 参数含义
- $\phi_i$：第 $i$ 阶滞后的条件影响。
- $p$：记忆长度；可结合 AIC/BIC 与 PACF 选择。
- $\varepsilon_t$：不可由过去预测的新冲击。
### 3.4 关键假设
| 假设 | 违反后果 | 粗查 |
| --- | --- | --- |
| 弱平稳 | 系数与预测区间失真 | 时序图、ADF/KPSS |
| 残差白噪声 | 尚有结构未建模 | 残差 ACF、Ljung–Box |
| 等间隔且无泄漏 | 滞后含义混乱 | 时间戳审计 |

## 4. 手把手算例
去均值 AR(1) 为 $y_t=0.6y_{t-1}+\varepsilon_t$，当前 $y_t=10$，未来冲击期望为 0：
$$
\hat y_{t+1|t}=0.6(10)=6,\qquad
\hat y_{t+2|t}=0.6(6)=3.6
$$
当前偏高 10 的影响按 $1,0.6,0.36,\ldots$ 衰减，而不是永久保持 10。

## 5. 数据形式与输入输出
### 5.1 适合的数据形式
单条等间隔连续序列；缺失需先合理处理，不直接处理删失或多患者层级相关。
### 5.2 示例表格
| date | admissions |
| --- | ---: |
| D1 | 20 |
| D2 | 23 |
| D3 | 22 |
### 5.3 输入与产出
输入序列和阶数 $p$；产出系数、残差、预测及预测区间。

## 6. 适用场景
适合平稳、线性、短记忆序列；趋势季节明显、结构突变或不规则采样时不宜直接使用。

## 7. 实现
### 7.1 Python
```python
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
rng = np.random.default_rng(42)
y = np.zeros(200)
for t in range(1, 200):
    y[t] = 0.6 * y[t-1] + rng.normal()
fit = AutoReg(y, lags=1, trend="c").fit()
print(fit.params, fit.predict(200, 204))
```
### 7.2 R
```r
set.seed(42)
y <- arima.sim(model = list(ar = 0.6), n = 200)
fit <- arima(y, order = c(1, 0, 0))
predict(fit, n.ahead = 5)
```

## 8. 结果如何解读
$\phi=0.6$ 表示控制更早滞后后，上期每高 1 单位，本期条件均值高 0.6；不是因果效应。

## 9. 假设诊断与稳健性
检查残差时序图、ACF、Ljung–Box、正态 QQ 图和滚动验证；异常点、变点与季节性需显式处理。

## 10. 推荐可视化
原序列、ACF/PACF、残差 ACF、滚动预测与区间。

## 11. 优势、局限与常见坑
优势是简单可解释；局限是线性和平稳。常见坑：随机切分、把相关当因果、差分不足、用同一数据选阶又报测试性能。

## 12. 与相近方法的区别
MA 建模过去冲击，AR 建模过去观测；ARMA 合并二者；非平稳序列先考虑 ARIMA。

## 13. 医学研究中的典型应用
短期床位需求、实验室质控、稳定采样生命体征；多患者重复测量不能直接当一条独立序列。

## 14. 关键术语
- **滞后（Lag）**：过去若干步的观测。
- **平稳性（Stationarity）**：均值、方差和协方差结构不随时间漂移。
- **PACF**：控制中间滞后后的直接相关。
- **白噪声（White noise）**：均值为零且无序列相关的新冲击。

## 15. 相关方法
- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]

## 16. 参考资料
- Box GEP, Jenkins GM, Reinsel GC, Ljung GM. *Time Series Analysis*. Wiley; 2015.
- Hamilton JD. *Time Series Analysis*. Princeton University Press; 1994.

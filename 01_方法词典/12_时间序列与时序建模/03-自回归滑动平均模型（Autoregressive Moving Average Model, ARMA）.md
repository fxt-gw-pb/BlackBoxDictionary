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
### 1.1 一句话本质
ARMA 同时用过去观测的惯性和过去冲击的回声，以较少参数描述平稳序列复杂自相关。
### 1.2 定义
ARMA($p,q$) 合并 AR($p$) 与 MA($q$)，适用于已平稳的单变量序列。
### 1.3 它主要解决什么问题
当 ACF/PACF 都拖尾、纯 AR 或纯 MA 需要很高阶时进行简约建模。
### 1.4 直觉与类比
序列既会沿用昨天的状态，也会继续消化昨天未预料到的冲击。

## 2. 核心思想与原理
### 2.1 根本困难
惯性和短期冲击常同时存在，单一机制会把另一机制误塞进高阶参数。
### 2.2 关键洞察
AR 产生持续衰减记忆，MA 产生有限冲击记忆；组合可生成更丰富但仍简约的相关结构。
### 2.3 与朴素做法对比
相比高阶 AR，低阶 ARMA 可能参数更少；代价是 MA 创新未观测、估计更复杂。

## 3. 数学形式
### 3.1 核心公式
$$
y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}
+\varepsilon_t+\sum_{j=1}^{q}\theta_j\varepsilon_{t-j}
$$
### 3.2 推导脉络
用滞后算子写为 $\phi(B)y_t=c+\theta(B)\varepsilon_t$；AR 多项式决定平稳性，MA 多项式决定可逆性。
### 3.3 参数含义
$p$ 是观测记忆阶数，$q$ 是冲击记忆阶数；AIC/BIC 与残差诊断共同选型。
### 3.4 关键假设
弱平稳、可逆、参数稳定、残差白噪声、等间隔；结构变点会使全期单一参数失真。

## 4. 手把手算例
ARMA(1,1)：$y_t=0.5y_{t-1}+\varepsilon_t+0.4\varepsilon_{t-1}$，$y_0=\varepsilon_0=0$。
若 $\varepsilon_1=2,\varepsilon_2=-1,\varepsilon_3=0$：
$$
y_1=2
$$
$$
y_2=0.5(2)-1+0.4(2)=0.8
$$
$$
y_3=0.5(0.8)+0+0.4(-1)=0
$$
第 2 期既继承 $y_1$ 的 1.0，又保留冲击 2 的 0.8，再叠加新冲击 $-1$。

## 5. 数据形式与输入输出
### 5.1 适合的数据形式
等间隔、平稳、连续单变量序列；不直接处理趋势、删失或层级数据。
### 5.2 示例表格
| time | y |
| --- | ---: |
| 1 | 2.0 |
| 2 | 0.8 |
| 3 | 0.0 |
### 5.3 输入与产出
输入序列与 $(p,q)$；产出系数、创新、信息准则和多步预测区间。

## 6. 适用场景
适合平稳且 ACF/PACF 均拖尾的序列；明显趋势季节、非线性或变点需扩展。

## 7. 实现
### 7.1 Python
```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
rng = np.random.default_rng(42)
e = rng.normal(size=300)
y = np.zeros(300)
for t in range(1, 300):
    y[t] = 0.5*y[t-1] + e[t] + 0.4*e[t-1]
fit = ARIMA(y, order=(1, 0, 1)).fit()
print(fit.summary(), fit.forecast(5))
```
### 7.2 R
```r
set.seed(42)
y <- arima.sim(model = list(ar = 0.5, ma = 0.4), n = 300)
fit <- arima(y, order = c(1, 0, 1))
predict(fit, n.ahead = 5)
```

## 8. 结果如何解读
$\phi$ 描述条件惯性，$\theta$ 描述旧创新回声；二者可能抵消，不能逐个脱离整体动态解释。

## 9. 假设诊断与稳健性
检查平稳/可逆根、残差 ACF 与 Ljung–Box、参数近边界、滚动验证和变点。

## 10. 推荐可视化
序列、ACF/PACF、残差诊断、滚动预测和冲击响应。

## 11. 优势、局限与常见坑
优势是简约灵活；局限是阶数识别与非线性。常见坑：只凭 ACF 机械选阶、搜索后不留测试集。

## 12. 与相近方法的区别
纯 AR 令 $q=0$，纯 MA 令 $p=0$；ARIMA 对差分后序列使用 ARMA。

## 13. 医学研究中的典型应用
平稳化后的床位、监护与实验室质控序列短期预测。

## 14. 关键术语
- **滞后算子（Backshift operator）**：$By_t=y_{t-1}$。
- **简约性（Parsimony）**：用尽量少参数解释动态。
- **信息准则**：拟合与参数数量的权衡。
- **拖尾（Tailing off）**：相关随阶数逐渐衰减而非突然归零。

## 15. 相关方法
- [[自回归模型（Autoregressive Model, AR）]]
- [[移动平均模型（Moving Average Model, MA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]

## 16. 参考资料
- Box GEP, et al. *Time Series Analysis*. Wiley; 2015.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. OTexts; 2021.

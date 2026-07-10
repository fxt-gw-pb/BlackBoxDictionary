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
### 1.1 一句话本质
统计时间序列中的 MA 用有限个过去随机冲击解释当前值，而不是对过去观测做滚动平均。
### 1.2 定义
MA($q$) 是白噪声及其前 $q$ 期值的线性组合，天然为平稳过程。
### 1.3 它主要解决什么问题
刻画一次异常冲击在有限时间内持续影响观测的序列。
### 1.4 直觉与类比
一次设备误差或突发事件会在当前及随后几期留下“回声”，到第 $q+1$ 期完全消失。

## 2. 核心思想与原理
### 2.1 根本困难
过去观测混合了许多冲击；MA 直接描述新冲击如何传播。
### 2.2 关键洞察
有限冲击记忆导致理论 ACF 在 $q$ 阶后截尾。
### 2.3 与朴素做法对比
滚动平均是平滑器；MA($q$) 是概率生成模型，参数、残差和预测区间均有统计含义。

## 3. 数学形式
### 3.1 核心公式
$$
y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
$$
这个式子在说：当前值由当前新冲击与有限个旧冲击共同组成。
### 3.2 推导脉络
把序列表示为白噪声的线性滤波；可逆性保证能由观测唯一恢复冲击，MA(1) 常要求 $|\theta|<1$。
### 3.3 参数含义
$\theta_j$ 是冲击第 $j$ 期回声；$q$ 是冲击记忆长度。
### 3.4 关键假设
平稳、可逆、白噪声残差、等间隔；用残差 ACF/Ljung–Box 和根检查。

## 4. 手把手算例
设 $\mu=0,\theta=0.5$，冲击依次为 $(2,-1,3,0)$，且 $\varepsilon_0=0$：
$$
y_1=2,\quad y_2=-1+0.5(2)=0
$$
$$
y_3=3+0.5(-1)=2.5,\quad y_4=0+0.5(3)=1.5
$$
冲击 3 在当期贡献 3、下一期贡献 1.5，之后归零，展示有限记忆。

## 5. 数据形式与输入输出
### 5.1 适合的数据形式
单变量等间隔平稳序列；不直接处理删失、趋势或多患者层级。
### 5.2 示例表格
| day | centered_count |
| --- | ---: |
| 1 | 2.0 |
| 2 | 0.0 |
| 3 | 2.5 |
### 5.3 输入与产出
输入序列和 $q$；产出 $\theta$、创新残差、预测与区间。

## 6. 适用场景
适合 ACF 明显截尾的短冲击过程；持续惯性更适合 AR，非平稳序列先差分。

## 7. 实现
### 7.1 Python
```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
rng = np.random.default_rng(42)
e = rng.normal(size=300)
y = e.copy()
y[1:] += 0.5 * e[:-1]
fit = ARIMA(y, order=(0, 0, 1), trend="c").fit()
print(fit.params, fit.forecast(5))
```
### 7.2 R
```r
set.seed(42)
y <- arima.sim(model = list(ma = 0.5), n = 300)
fit <- arima(y, order = c(0, 0, 1))
predict(fit, n.ahead = 5)
```

## 8. 结果如何解读
$\theta_1=0.5$ 表示一次正冲击下一期仍留下其一半回声；符号约定在不同软件中可能相反。

## 9. 假设诊断与稳健性
看 ACF 截尾、残差白噪声、可逆根、异常冲击和滚动预测；不要从估计残差中作因果解释。

## 10. 推荐可视化
序列、ACF/PACF、冲击响应、残差 ACF、预测区间。

## 11. 优势、局限与常见坑
优势是有限记忆清晰；局限是冲击不可直接观测。常见坑是把 MA 模型与 rolling mean 混为一谈。

## 12. 与相近方法的区别
AR 用过去值；MA 用过去创新；ARMA 同时使用；指数平滑是另一类预测递推。

## 13. 医学研究中的典型应用
设备误差回声、短期报告延迟、突发事件后的有限持续影响。

## 14. 关键术语
- **创新（Innovation）**：看到过去后仍不可预测的新信息。
- **可逆性（Invertibility）**：观测可唯一对应稳定创新表示。
- **ACF 截尾**：超过某阶理论自相关为零。
- **滚动平均**：对观测求局部均值的平滑操作，不是 MA 模型。

## 15. 相关方法
- [[自回归模型（Autoregressive Model, AR）]]
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]]

## 16. 参考资料
- Box GEP, et al. *Time Series Analysis*. Wiley; 2015.
- Brockwell PJ, Davis RA. *Introduction to Time Series and Forecasting*. Springer; 2016.

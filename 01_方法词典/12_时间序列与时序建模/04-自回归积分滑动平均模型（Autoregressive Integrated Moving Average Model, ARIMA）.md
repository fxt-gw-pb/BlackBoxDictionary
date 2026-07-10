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
### 1.1 一句话本质
ARIMA 先用差分去掉随机趋势，再对差分后的平稳变化量建立 ARMA。
### 1.2 定义
ARIMA($p,d,q$) 中 $d$ 是差分次数，$p,q$ 是差分序列的 AR/MA 阶数。
### 1.3 它主要解决什么问题
带趋势但可经有限差分平稳化的单变量序列预测。
### 1.4 直觉与类比
不预测不断上升的“高度”，先预测每期“增量”，再把增量累加回水平。

## 2. 核心思想与原理
### 2.1 根本困难
非平稳水平会产生伪相关，固定均值的 ARMA 不适用。
### 2.2 关键洞察
差分算子 $(1-B)^d$ 可移除单位根型趋势；预测后再积分还原原尺度。
### 2.3 与朴素做法对比
线性趋势外推假定确定性趋势；ARIMA 允许随机趋势及相关增量。

## 3. 数学形式
### 3.1 核心公式
$$
\phi(B)(1-B)^d y_t=c+\theta(B)\varepsilon_t
$$
这个式子在说：对 $y_t$ 做 $d$ 次差分后，剩余序列服从 ARMA($p,q$)。
### 3.2 推导脉络
先判断差分需求，再用 ACF/PACF、AIC/BIC 选择 $p,q$，最后检查残差；过度差分会引入额外负相关并放大噪声。
### 3.3 参数含义
$d$ 是积分阶数；drift 是差分均值；季节性需扩展为 SARIMA。
### 3.4 关键假设
差分后平稳、残差白噪声、参数稳定、等间隔；结构变点和季节性必须另建模。

## 4. 手把手算例
水平序列为 $(100,103,107,108,112)$，一阶差分：
$$
\Delta y=(3,4,1,4)
$$
若采用 ARIMA(0,1,0) with drift，增量均值为
$$
\bar\Delta=(3+4+1+4)/4=3
$$
于是未来两期：
$$
\hat y_6=112+3=115,\qquad
\hat y_7=115+3=118
$$
差分域预测的是 3、3；“积分”就是把它们逐期累加回原尺度。

## 5. 数据形式与输入输出
### 5.1 适合的数据形式
等间隔单变量连续序列；缺失、季节、干预和删失需专门处理。
### 5.2 示例表格
| month | visits |
| --- | ---: |
| 1 | 100 |
| 2 | 103 |
| 3 | 107 |
### 5.3 输入与产出
输入序列及 $(p,d,q)$；产出差分模型、原尺度预测和随期数扩大的区间。

## 6. 适用场景
适合单位根型趋势和短中期预测；复杂季节、多个协变量、非线性或结构突变需扩展。

## 7. 实现
### 7.1 Python
```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
y = np.array([100, 103, 107, 108, 112], dtype=float)
fit = ARIMA(y, order=(0, 1, 0), trend="t").fit()
print(fit.forecast(2))
```
### 7.2 R
```r
y <- ts(c(100, 103, 107, 108, 112))
fit <- arima(y, order = c(0, 1, 0), include.drift = TRUE)
predict(fit, n.ahead = 2)
```

## 8. 结果如何解读
参数属于差分序列；预测需回到原尺度。远期区间变宽反映增量误差不断累积。

## 9. 假设诊断与稳健性
比较 $d=0,1,2$，做 ADF/KPSS、残差 Ljung–Box、滚动起点验证和变点/季节图。

## 10. 推荐可视化
原序列与差分序列、ACF/PACF、残差、原尺度预测区间。

## 11. 优势、局限与常见坑
优势是经典透明；局限是线性且对结构变化敏感。常见坑：过度差分、随机切分、把 auto-ARIMA 当无需诊断。

## 12. 与相近方法的区别
ARMA 要求原序列平稳；ARIMA 允许差分；SARIMA 加季节差分；指数平滑从状态递推角度建模趋势季节。

## 13. 医学研究中的典型应用
月度门诊量、药品消耗、疾病监测计数经适当变换后的预测；疫情干预需显式加入干预项。

## 14. 关键术语
- **差分（Differencing）**：$\Delta y_t=y_t-y_{t-1}$。
- **单位根（Unit root）**：冲击永久进入水平的非平稳结构。
- **积分（Integrated）**：差分后平稳，预测再累加回水平。
- **Drift**：一阶差分的非零均值。

## 15. 相关方法
- [[自回归滑动平均模型（Autoregressive Moving Average Model, ARMA）]]
- [[Holt-Winters指数平滑（Holt-Winters Exponential Smoothing）]]
- [[Prophet时间序列模型（Prophet Forecasting Model）]]

## 16. 参考资料
- Box GEP, et al. *Time Series Analysis*. Wiley; 2015.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. OTexts; 2021.

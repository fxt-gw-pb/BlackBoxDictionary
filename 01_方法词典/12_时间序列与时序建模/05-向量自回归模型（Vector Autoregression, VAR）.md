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
### 1.1 一句话本质
VAR 让一组变量都由这组变量的过去共同预测，从而刻画相互滞后联动。
### 1.2 定义
VAR($p$) 是多变量 AR：每个方程使用所有变量的 $p$ 阶滞后。
### 1.3 它主要解决什么问题
多条平稳时间序列的联合预测、Granger 预测关系、脉冲响应与方差分解。
### 1.4 直觉与类比
心率不仅延续自身惯性，也可能跟随先前乳酸；乳酸也可能受先前心率影响。

## 2. 核心思想与原理
### 2.1 根本困难
逐变量建 AR 会遗漏跨变量的动态信息，并使残差相关。
### 2.2 关键洞察
把多个序列堆成向量，以系数矩阵统一描述“谁的过去预测谁的现在”。
### 2.3 与朴素做法对比
VAR 少设结构、适合预测；代价是参数按变量数平方增长，结构性因果识别需额外假设。

## 3. 数学形式
### 3.1 核心公式
$$
\mathbf y_t=\mathbf c+\mathbf A_1\mathbf y_{t-1}
+\cdots+\mathbf A_p\mathbf y_{t-p}+\boldsymbol\varepsilon_t
$$
### 3.2 推导脉络
每个方程可用相同解释变量分别 OLS；联合动态由 companion matrix 特征根决定。
### 3.3 参数含义
$A_{k,ij}$ 表示控制其他滞后后，变量 $j$ 的第 $k$ 阶滞后对变量 $i$ 的预测关系。
### 3.4 关键假设
联合平稳、参数稳定、残差序列不相关；脉冲响应还需对同期残差相关作识别。

## 4. 手把手算例
设 VAR(1)：
$$
\mathbf A=\begin{pmatrix}0.5&0.2\\0.1&0.6\end{pmatrix},
\qquad \mathbf y_t=(2,1)^\top
$$
无未来冲击时：
$$
\hat{\mathbf y}_{t+1}=\mathbf A\mathbf y_t=(1.2,0.8)^\top
$$
$$
\hat{\mathbf y}_{t+2}=\mathbf A(1.2,0.8)^\top=(0.76,0.60)^\top
$$
若第一个变量受到单位冲击，脉冲响应为
$$
h=0:(1,0),\quad h=1:(0.5,0.1),\quad h=2:(0.27,0.11)
$$
冲击不仅自身衰减，也通过交叉系数传到第二变量。

## 5. 数据形式与输入输出
### 5.1 适合的数据形式
同频率、对齐的多变量连续平稳序列；变量过多会参数爆炸。
### 5.2 示例表格
| hour | heart_rate_z | lactate_z |
| --- | ---: | ---: |
| 1 | 2.0 | 1.0 |
| 2 | 1.2 | 0.8 |
### 5.3 输入与产出
输入对齐矩阵和 $p$；产出系数矩阵、联合预测、Granger 检验、IRF 与 FEVD。

## 6. 适用场景
适合少量相互影响的平稳序列；高维、小样本、协整或不规则采样需正则 VAR/VECM 等方法。

## 7. 实现
### 7.1 Python
```python
import numpy as np
from statsmodels.tsa.api import VAR
rng = np.random.default_rng(42)
A = np.array([[0.5, 0.2], [0.1, 0.6]])
y = np.zeros((300, 2))
for t in range(1, 300):
    y[t] = A @ y[t-1] + rng.normal(size=2)
fit = VAR(y).fit(1)
print(fit.forecast(y[-1:], steps=3))
```
### 7.2 R
```r
library(vars)
set.seed(42)
A <- matrix(c(0.5, 0.1, 0.2, 0.6), 2, 2)
y <- matrix(0, 300, 2)
for (t in 2:300) y[t, ] <- A %*% y[t-1, ] + rnorm(2)
fit <- VAR(as.data.frame(y), p = 1, type = "const")
predict(fit, n.ahead = 3)
```

## 8. 结果如何解读
系数是条件预测关系，不自动等于因果效应；Granger 因果仅表示过去值增加预测信息。

## 9. 假设诊断与稳健性
检查联合平稳性、残差自相关/协方差、根、滞后阶、变量顺序对正交 IRF 的敏感性和滚动验证。

## 10. 推荐可视化
多序列标准化轨迹、系数热图、IRF 置信带、FEVD 堆积图、滚动预测。

## 11. 优势、局限与常见坑
优势是对称联合建模；局限是参数多、识别弱。常见坑：量纲未处理、变量过多、把 Granger 当机制因果。

## 12. 与相近方法的区别
AR 是单变量特例；VARMA 还建模冲击滞后；协整非平稳序列应用 VECM；外生变量可用 VARX。

## 13. 医学研究中的典型应用
多项生命体征联动、医院资源序列、公共卫生指标动态；单个患者间差异与信息性缺失需另建模。

## 14. 关键术语
- **Granger 因果**：过去信息是否改善预测，不是机制因果。
- **脉冲响应（IRF）**：单位冲击对未来各变量的动态影响。
- **FEVD**：预测误差由各冲击贡献的比例。
- **同期识别**：把相关残差分解为结构冲击所需的额外假设。

## 15. 相关方法
- [[自回归模型（Autoregressive Model, AR）]]
- [[向量自回归滑动平均模型（Vector Autoregressive Moving Average Model, VARMA）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]

## 16. 参考资料
- Lütkepohl H. *New Introduction to Multiple Time Series Analysis*. Springer; 2005.
- Hamilton JD. *Time Series Analysis*. Princeton University Press; 1994.

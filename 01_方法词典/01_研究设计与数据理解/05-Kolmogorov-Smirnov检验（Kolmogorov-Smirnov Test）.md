---
title: Kolmogorov-Smirnov检验
english_name: Kolmogorov-Smirnov Test
slug: kolmogorov-smirnov-test
aliases: [KS test, K-S检验, Kolmogorov-Smirnov, "Kolmogorov-Smirnov检验（Kolmogorov-Smirnov Test）"]
category: 研究设计与数据理解
subcategory: 分布检验
tags: [医学统计, 数据科学, 非参数检验, 分布拟合]
status: 已建
difficulty: intermediate
question_type: 分布拟合优度与两样本分布比较
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scipy]
r_packages: [stats]
---

# Kolmogorov-Smirnov检验（Kolmogorov-Smirnov Test）

## 1. 方法概览

### 1.1 定义

Kolmogorov-Smirnov 检验是一种基于经验分布函数的非参数检验，用经验分布与理论分布（单样本）或两组经验分布之间的最大距离，判断分布是否一致。

### 1.2 它主要解决什么问题

- 研究问题：一份数据是否服从某个指定分布，或两组数据是否来自同一分布。
- 适用任务：分布拟合优度检验、两样本分布比较、正态性初查。
- 常见医学场景：检查指标是否近似正态、比较两组连续指标的整体分布差异。

### 1.3 直觉理解

把数据画成经验分布函数（ECDF）阶梯曲线，KS 统计量就是这条曲线与参照曲线之间“上下张开”的最大缝隙。缝隙越大，越有理由认为两条分布不同。

## 2. 数学形式

### 2.1 核心公式

单样本：经验分布 $F_n(x)$ 与理论分布 $F_0(x)$ 的最大绝对差；两样本：两经验分布之差：

$$
D_n=\sup_x\,|F_n(x)-F_0(x)|,\qquad
D_{m,n}=\sup_x\,|F_{1,m}(x)-F_{2,n}(x)|
$$

大样本下 $\sqrt{n}\,D_n$ 收敛到 Kolmogorov 分布，用于计算 p 值。

### 2.2 参数或统计量含义

- $F_n(x)$：样本的经验分布函数。
- $D$：最大分布距离，即 KS 统计量。
- p 值：在原假设（分布相同）下观察到该距离的概率。
- 对分布中部差异较敏感，对尾部相对不敏感。

### 2.3 关键假设

- 变量为连续型；参照分布若含估计参数，需用校正版本（如 Lilliefors）。
- 观测独立同分布。
- 结点（重复值）会削弱检验，理论基于连续分布。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单个连续变量（单样本）或两组连续变量（两样本）。
- 因变量类型：不区分。
- 数据结构：一维数值数组或两数组。
- 是否适合高维数据：为一维检验；多维需专门扩展。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接适用。
- 是否适合重复测量数据：独立性破坏时不适用。

### 3.2 示例表格

| 检验类型 | 输入 | 原假设 |
| --- | --- | --- |
| 单样本 | 数据 + 理论分布 | 数据服从该分布 |
| 两样本 | 两组数据 | 两组同分布 |

### 3.3 输入与产出

#### 输入

- 输入数据：一维数值向量（及参照分布）。
- 关键变量：待检数据、参照分布或第二组数据。
- 需要预处理的内容：确认连续、避免用同一数据估参再检验（否则需校正）。

#### 产出

- 模型对象/统计结果：KS 统计量 $D$、p 值。
- 参数估计：无。
- 预测结果：无。
- 不确定性指标：p 值。

## 4. 适用场景

- 适合：连续变量的分布拟合优度、两组整体分布比较。
- 不适合：小样本（功效低）、对尾部差异敏感的问题、离散/多结点数据。
- 使用前需要特别检查的点：参照分布参数是否来自同一数据、结点比例、样本量。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`

```python
from scipy import stats
import numpy as np

x = np.random.normal(0, 1, 200)
# 单样本: 与标准正态比较
print(stats.kstest(x, "norm"))
# 两样本
y = np.random.normal(0.3, 1, 200)
print(stats.ks_2samp(x, y))
```

### 5.2 R

常用包：

- `stats`

```r
x <- rnorm(200)
ks.test(x, "pnorm")          # 单样本
y <- rnorm(200, 0.3)
ks.test(x, y)                # 两样本
```

## 6. 结果如何解释

- 核心结果看什么：$D$ 的大小与 p 值。
- 每个主要参数如何解释：p 小说明有证据认为分布不同；$D$ 反映最大分布偏离处。
- 临床或医学意义如何表达：拒绝正态性提示后续应考虑非参数方法或变换。
- 常见误读：不拒绝≠证明分布相同；大样本下极小差异也会显著。

## 7. 推荐可视化

- 两条 ECDF 叠加图并标出最大缝隙 $D$。
- 与直方图/QQ 图配合判断分布。
- 两样本 ECDF 差异曲线。

## 8. 优势、局限与常见坑

### 优势

- 非参数、无需假设具体分布族。
- 对整体分布差异敏感，直观（基于 ECDF）。
- 有单样本与两样本两种形式。

### 局限

- 小样本功效低。
- 对尾部差异不如中部敏感。
- 参数由数据估计时标准版会过于保守，需校正。

### 常见坑

- 用同一数据估计正态参数后直接 ks.test(x,"pnorm")（应用 Lilliefors）。
- 把“不显著”当作“服从该分布”。
- 忽略结点对连续假设的破坏。

## 9. 与相近方法的区别

- 和 [[经验分布函数（Empirical Cumulative Distribution Function, ECDF）]] 的区别：ECDF 是描述工具，KS 在其上做假设检验。
- 和 Shapiro-Wilk 正态性检验的区别：Shapiro-Wilk 专为正态性、小样本功效更高；KS 更通用但功效较低。
- 和 [[Wilcoxon秩和检验（Wilcoxon Rank-Sum Test）]] 的区别：后者聚焦位置差异，KS 检验整体分布差异。

## 10. 医学研究中的典型应用

- 建模前检查连续变量是否近似正态。
- 比较两组患者某指标的整体分布是否不同。
- 校验模拟/插补数据与原始分布的一致性。

## 11. 相关方法

- [[经验分布函数（Empirical Cumulative Distribution Function, ECDF）]]
- [[核密度估计（Kernel Density Estimation, KDE）]]
- [[Wilcoxon秩和检验（Wilcoxon Rank-Sum Test）]]

## 12. 参考资料

- Massey FJ. The Kolmogorov-Smirnov test for goodness of fit. *J Am Stat Assoc*. 1951;46(253):68-78.
- Conover WJ. *Practical Nonparametric Statistics*. 3rd ed. Wiley; 1999.
- R Core Team. `ks.test`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html) （访问日期：2026-07-02）

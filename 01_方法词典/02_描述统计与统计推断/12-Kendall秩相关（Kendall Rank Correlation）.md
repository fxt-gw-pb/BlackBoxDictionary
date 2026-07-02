---
title: Kendall秩相关
english_name: Kendall Rank Correlation
slug: kendall-rank-correlation
aliases: [Kendall tau, Kendall秩相关系数, "Kendall秩相关（Kendall Rank Correlation）"]
category: 描述统计与统计推断
subcategory: 相关分析
tags: [医学统计, 数据科学, 相关分析, 非参数, 秩相关]
status: 已建
difficulty: basic
question_type: 单调相关分析
data_type: [表格数据]
outcome_type: [连续型, 等级]
python_packages: [scipy]
r_packages: [stats]
---

# Kendall秩相关（Kendall Rank Correlation）

## 1. 方法概览

### 1.1 定义

Kendall 秩相关系数（$\tau$）通过比较所有样本对的“一致”与“不一致”，衡量两个变量之间单调关系的方向与强度，是一种非参数相关。

### 1.2 它主要解决什么问题

- 研究问题：两个变量是否单调同向变化，不假设线性或正态。
- 适用任务：序数变量相关、小样本或多结点数据的稳健相关、评分者一致性。
- 常见医学场景：疾病分级与另一序数指标的关联、量表得分相关、小样本关联分析。

### 1.3 直觉理解

任取两名个体，如果在两个变量上排序方向一致就记“一致对”，相反就记“不一致对”。$\tau$ 就是一致对减去不一致对占全部对的比例，直接刻画“同向排序”的净倾向。

## 2. 数学形式

### 2.1 核心公式

设一致对数为 $C$、不一致对数为 $D$，样本量为 $n$：

$$
\tau_a=\frac{C-D}{\binom{n}{2}}
$$

存在结点时用 $\tau_b$ 校正：

$$
\tau_b=\frac{C-D}{\sqrt{(n_0-n_1)(n_0-n_2)}}
$$

其中 $n_0=\binom{n}{2}$，$n_1,n_2$ 为两变量内结点带来的对数校正项。

### 2.2 参数或统计量含义

- $C,D$：一致对、不一致对数。
- $\tau_a$：无结点版本；$\tau_b$：有结点时的校正版本。
- $\tau$ 值：范围 $[-1,1]$，可解释为一致概率与不一致概率之差。

### 2.3 关键假设

- 变量至少为序数，关系为单调即可（不需线性/正态）。
- 观测成对且近似独立。
- 对离群值稳健，但极端结点仍需用 $\tau_b$ 处理。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续或序数。
- 因变量类型：连续或序数。
- 数据结构：成对观测。
- 是否适合高维数据：可两两计算，但成本高于 Pearson。
- 是否适合缺失较多数据：需成对完整。
- 是否适合删失数据：不直接适用。
- 是否适合重复测量数据：独立性破坏时需谨慎。

### 3.2 示例表格

| 个体 | 疾病分级(1-4) | 生活质量评分(序数) |
| --- | --- | --- |
| 1 | 1 | 5 |
| 2 | 2 | 4 |
| 3 | 3 | 4 |
| 4 | 4 | 2 |

### 3.3 输入与产出

#### 输入

- 输入数据：两个成对的连续/序数变量。
- 关键变量：$x$、$y$。
- 需要预处理的内容：确认为序数或单调关系；结点较多时选 $\tau_b$。

#### 产出

- 模型对象/统计结果：$\tau$、p 值。
- 参数估计：$\tau$ 点估计。
- 预测结果：无。
- 不确定性指标：p 值、（可选）bootstrap 区间。

## 4. 适用场景

- 适合：序数数据、小样本、多结点、需要稳健单调相关的场景。
- 不适合：需要线性强度度量、连续正态数据下追求效率（此时 Pearson 更有效）。
- 使用前需要特别检查的点：是否为单调关系、结点比例、样本独立性。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`

```python
from scipy import stats

tau, p = stats.kendalltau(df["grade"], df["qol"])
print(f"tau={tau:.3f}, p={p:.4f}")
```

### 5.2 R

常用包：

- `stats`

```r
cor.test(df$grade, df$qol, method = "kendall")
```

## 6. 结果如何解释

- 核心结果看什么：$\tau$ 的方向与大小、p 值。
- 每个主要参数如何解释：$\tau=0.4$ 表示一致对比不一致对多出约 40 个百分点，单调正相关中等。
- 临床或医学意义如何表达：适合描述“分级越高，另一指标越倾向变差/变好”的单调趋势。
- 常见误读：把 $\tau$ 与 Spearman 的 $\rho$ 直接数值互换（二者尺度不同，$\tau$ 通常偏小）。

## 7. 推荐可视化

- 分级散点或抖动图配趋势。
- 有序分类的热图/马赛克图。
- 与 Spearman、Pearson 的对照条形。

## 8. 优势、局限与常见坑

### 优势

- 对离群值稳健，不假设分布。
- 小样本、多结点时比 Spearman 更稳健，有直接的概率解释。
- 适合序数数据。

### 局限

- 计算比 Pearson/Spearman 略慢（大样本）。
- 只反映单调关系。
- 连续正态数据下效率低于 Pearson。

### 常见坑

- 有结点却用 $\tau_a$。
- 把 $\tau$ 与 $\rho$ 混为一谈。
- 用于明显非单调关系。

## 9. 与相近方法的区别

- 和 [[Spearman秩相关（Spearman Rank Correlation）]] 的区别：Spearman 是秩的 Pearson 相关，Kendall 基于对比对；Kendall 在小样本/多结点更稳健但值偏小。
- 和 [[Pearson相关（Pearson Correlation）]] 的区别：Pearson 度量线性、需正态，Kendall 度量单调、非参数。

## 10. 医学研究中的典型应用

- 疾病严重程度分级与序数临床指标的关联。
- 量表条目/评分的单调相关。
- 小样本探索性关联分析。

## 11. 相关方法

- [[Spearman秩相关（Spearman Rank Correlation）]]
- [[Pearson相关（Pearson Correlation）]]

## 12. 参考资料

- Kendall MG. A new measure of rank correlation. *Biometrika*. 1938;30(1-2):81-93.
- Hollander M, Wolfe DA, Chicken E. *Nonparametric Statistical Methods*. 3rd ed. Wiley; 2013.
- R Core Team. `cor.test`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html) （访问日期：2026-07-02）

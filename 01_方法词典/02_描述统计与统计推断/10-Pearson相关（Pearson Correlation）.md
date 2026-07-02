---
title: Pearson相关
english_name: Pearson Correlation
slug: pearson-correlation
aliases: [Pearson correlation, Pearson相关系数, r, "Pearson相关（Pearson Correlation）"]
category: 描述统计与统计推断
subcategory: 相关分析
tags: [医学统计, 数据科学, 相关分析, 线性关系]
status: 已建
difficulty: basic
question_type: 线性相关分析
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scipy]
r_packages: [stats]
---

# Pearson相关（Pearson Correlation）

## 1. 方法概览

### 1.1 定义

Pearson 相关系数衡量两个连续变量之间线性关系的方向与强度，取值在 −1 到 1 之间。

### 1.2 它主要解决什么问题

- 研究问题：两个连续指标是否同增同减，线性关联有多强。
- 适用任务：线性相关强度度量、变量筛选前的初步关联探索。
- 常见医学场景：BMI 与血压、剂量与生物标志物浓度、两次测量的一致性初探。

### 1.3 直觉理解

把两个变量都标准化后，Pearson 相关就是它们的“协同波动”程度：一起偏离均值时为正，反向偏离为负。它只捕捉直线关系，弯曲关系可能被严重低估。

## 2. 数学形式

### 2.1 核心公式

$$
r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

显著性检验统计量与 Fisher z 变换（用于置信区间）：

$$
t=\frac{r\sqrt{n-2}}{\sqrt{1-r^2}}\sim t_{n-2},\qquad z=\tfrac{1}{2}\ln\frac{1+r}{1-r}
$$

### 2.2 参数或统计量含义

- $r$：相关系数，符号表方向、绝对值表强度。
- $r^2$：决定系数，可解释的方差比例。
- $t$ 统计量：检验总体相关是否为 0。
- Fisher $z$：使抽样分布近似正态，用于构造 CI。

### 2.3 关键假设

- 两变量近似为区间/连续量，关系近似线性。
- 推断（p 值、CI）依赖双变量近似正态且方差有限。
- 对离群值敏感；无极端异常点或已妥善处理。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量。
- 因变量类型：连续变量（相关无因果方向之分）。
- 数据结构：成对观测，每行含两个数值。
- 是否适合高维数据：两两相关可扩展为相关矩阵。
- 是否适合缺失较多数据：需成对完整；缺失多时结果不稳。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：独立性被破坏时需专门方法（如重复测量相关）。

### 3.2 示例表格

| 个体 | BMI | 收缩压 |
| --- | --- | --- |
| 1 | 22.4 | 118 |
| 2 | 27.9 | 134 |
| 3 | 31.2 | 145 |
| 4 | 24.1 | 122 |

### 3.3 输入与产出

#### 输入

- 输入数据：两个成对的连续变量。
- 关键变量：$x$、$y$。
- 需要预处理的内容：检查线性、离群值、必要时变换。

#### 产出

- 模型对象/统计结果：$r$、p 值、置信区间。
- 参数估计：相关系数点估计。
- 预测结果：无（相关不预测）。
- 不确定性指标：Fisher z 区间、p 值。

## 4. 适用场景

- 适合：连续变量间的线性关联度量与初步探索。
- 不适合：非线性关系、等级/序数数据、存在强离群值、需要因果解释。
- 使用前需要特别检查的点：散点图形态、离群值、是否存在混杂导致的伪相关。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`

```python
from scipy import stats

r, p = stats.pearsonr(df["BMI"], df["SBP"])
print(f"r={r:.3f}, p={p:.4f}")
```

### 5.2 R

常用包：

- `stats`

```r
cor.test(df$BMI, df$SBP, method = "pearson")
```

## 6. 结果如何解释

- 核心结果看什么：$r$ 的方向与大小、置信区间、p 值。
- 每个主要参数如何解释：如 $r=0.6$ 表示较强正线性关联，$r^2=0.36$ 表示解释了约 36% 的方差。
- 临床或医学意义如何表达：相关不等于因果；强相关也可能来自共同混杂。
- 常见误读：把相关当因果；忽略离群值制造的假相关；用 Pearson 描述明显非线性的关系。

## 7. 推荐可视化

- 散点图叠加线性拟合线与置信带。
- 多变量相关矩阵热图。
- 离群值诊断（如带标注的散点）。

## 8. 优势、局限与常见坑

### 优势

- 概念直观、计算简单、可解释性强。
- 有成熟的检验与区间理论。
- 易扩展为相关矩阵用于探索。

### 局限

- 只捕捉线性关系。
- 对离群值高度敏感。
- 受测量误差衰减、范围受限影响。

### 常见坑

- 未看散点图直接报 $r$。
- 用于序数或明显偏态数据（应转 [[Spearman秩相关（Spearman Rank Correlation）]] 或 [[Kendall秩相关（Kendall Rank Correlation）]]）。
- 把相关系数当效应量做因果推断。

## 9. 与相近方法的区别

- 和 [[Spearman秩相关（Spearman Rank Correlation）]] 的区别：Spearman 基于秩，捕捉单调关系、对离群值更稳健。
- 和 [[Kendall秩相关（Kendall Rank Correlation）]] 的区别：Kendall 基于一致/不一致对，小样本与多结点时更稳健。
- 和 [[线性回归（Linear Regression）]] 的区别：简单线性回归的标准化斜率即 $r$，但回归有因变量方向并可多元调整。

## 10. 医学研究中的典型应用

- 连续生理指标间关联的初步描述。
- 生物标志物与临床连续结局的关联探索。
- 建模前的共线性与关联结构检查。

## 11. 相关方法

- [[Spearman秩相关（Spearman Rank Correlation）]]
- [[Kendall秩相关（Kendall Rank Correlation）]]
- [[线性回归（Linear Regression）]]

## 12. 参考资料

- Altman DG. *Practical Statistics for Medical Research*. Chapman and Hall/CRC; 1990.
- Bland M. *An Introduction to Medical Statistics*. 4th ed. Oxford University Press; 2015.
- R Core Team. `cor.test`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html) （访问日期：2026-07-02）

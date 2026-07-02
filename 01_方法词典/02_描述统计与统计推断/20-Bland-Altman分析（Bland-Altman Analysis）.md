---
title: Bland-Altman分析
english_name: Bland-Altman Analysis
slug: bland-altman-analysis
aliases: [Bland-Altman, 一致性界限, limits of agreement, "Bland-Altman分析（Bland-Altman Analysis）"]
category: 描述统计与统计推断
subcategory: 方法比较
tags: [医学统计, 数据科学, 方法比较, 一致性, 测量]
status: 已建
difficulty: basic
question_type: 两种测量方法的一致性评估
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [numpy, matplotlib]
r_packages: [blandr]
---

# Bland-Altman分析（Bland-Altman Analysis）

## 1. 方法概览

### 1.1 定义

Bland-Altman 分析通过绘制两种测量方法的「差值对均值」散点图，评估它们是否可以互换：给出平均偏差和 95% 一致性界限（limits of agreement）。

### 1.2 它主要解决什么问题

- 研究问题：一种新（便宜/无创）测量方法能否替代金标准方法。
- 适用任务：方法比较、仪器校验、测量互换性评估。
- 常见医学场景：新血压计对比动脉导管、无创血氧对比动脉血气、两种影像测量。

### 1.3 直觉理解

相关系数高不等于两法可互换——两法可以高度相关却系统性地相差很多。Bland-Altman 直接看「两法差多少」，只要差异的大小在临床可接受范围内，两法才算可互换。

## 2. 数学形式

### 2.1 核心公式

对每个对象计算两法差值 $d_i=x_{1i}-x_{2i}$，偏差与一致性界限为：

$$
\bar d=\frac{1}{n}\sum_i d_i,\qquad
\text{LoA}=\bar d \pm 1.96\,s_d
$$

### 2.2 参数或统计量含义

- $\bar d$：平均偏差（bias），系统性差异。
- $s_d$：差值的标准差。
- LoA：95% 一致性界限，约 95% 的个体差值落入其中。
- 可对 $\bar d$ 与 LoA 各自给置信区间。

### 2.3 关键假设

- 差值近似正态、且方差不随测量水平变化（否则需对数变换或回归型 LoA）。
- 配对测量、对象独立。
- 「可接受范围」应由临床事先设定，而非由数据决定。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：无，两法配对连续测量。
- 因变量类型：连续测量值。
- 数据结构：每行一个对象、两列测量。
- 是否适合高维数据：单指标比较。
- 是否适合缺失较多数据：需成对完整。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：每对象多次测量需用重复测量版 LoA。

### 3.2 示例表格

| Subject | 金标准 | 新方法 | 差值 |
| --- | --- | --- | --- |
| 1 | 120 | 118 | 2 |
| 2 | 135 | 140 | -5 |
| 3 | 98 | 96 | 2 |
| 4 | 150 | 152 | -2 |

### 3.3 输入与产出

#### 输入

- 输入数据：两法配对测量。
- 关键变量：两测量列、临床可接受差异阈值。
- 需要预处理的内容：检查差值正态性与方差齐性，必要时变换。

#### 产出

- 模型对象/统计结果：偏差、$s_d$、95% LoA 及其区间。
- 参数估计：$\bar d$、LoA。
- 预测结果：无。
- 不确定性指标：偏差与 LoA 的置信区间。

## 4. 适用场景

- 适合：判断两种连续测量方法能否互换。
- 不适合：评估单一测量的信度（用 [[组内相关系数（Intraclass Correlation Coefficient, ICC）]]）、分类判断一致性（用 [[Cohen Kappa一致性系数（Cohen's Kappa）]]）。
- 使用前需要特别检查的点：差值是否随均值增大而变大（比例误差）、是否需事先设定可接受界限。

## 5. 实现

### 5.1 Python

常用包：

- `numpy`
- `matplotlib`

```python
import numpy as np
import matplotlib.pyplot as plt

m1 = np.array([120,135,98,150,110])
m2 = np.array([118,140,96,152,108])
mean = (m1+m2)/2; diff = m1-m2
bias, sd = diff.mean(), diff.std(ddof=1)
loa = (bias-1.96*sd, bias+1.96*sd)

plt.scatter(mean, diff)
for y in (bias, *loa): plt.axhline(y, ls="--")
plt.xlabel("两法均值"); plt.ylabel("差值"); plt.show()
print("bias=%.2f, LoA=%.2f~%.2f" % (bias, *loa))
```

### 5.2 R

常用包：

- `blandr`

```r
library(blandr)
m1 <- c(120,135,98,150,110)
m2 <- c(118,140,96,152,108)
blandr.draw(m1, m2)
blandr.statistics(m1, m2)$bias
```

## 6. 结果如何解释

- 核心结果看什么：偏差是否接近 0、LoA 宽度是否在临床可接受范围内。
- 每个主要参数如何解释：偏差是系统差异，LoA 反映个体层面的最大可能差异。
- 临床或医学意义如何表达：若 LoA 完全落在预设可接受区间内，两法可互换。
- 常见误读：用高相关系数论证可互换；把 LoA 与置信区间混淆。

## 7. 推荐可视化

- Bland-Altman 图（差值 vs 均值，标注偏差与 LoA）。
- 差值直方图/QQ 图检验正态性。
- 若有比例误差，画差值对均值的回归型 LoA。

## 8. 优势、局限与常见坑

### 优势

- 直观区分系统偏差与随机差异。
- 直接对接临床可接受性判断。
- 计算与展示简单。

### 局限

- 需事先设定可接受界限，方法本身不给判据。
- 差值非正态或有比例误差时需变换/回归型 LoA。
- 重复测量需专门方法，否则 LoA 偏窄。

### 常见坑

- 用相关系数或回归代替一致性评估。
- 忽视比例误差（差值随均值增大）。
- 样本量太小导致 LoA 区间很宽却不报告。

## 9. 与相近方法的区别

- 和 [[Pearson相关（Pearson Correlation）]] 的区别：相关看共变，Bland-Altman 看差异大小；高相关不代表可互换。
- 和 [[组内相关系数（Intraclass Correlation Coefficient, ICC）]] 的区别：ICC 给单一信度指数，Bland-Altman 展示偏差与量程依赖。
- 和回归/Deming 回归的区别：回归估计换算关系，Bland-Altman 评估互换性。

## 10. 医学研究中的典型应用

- 无创与有创血压/血氧测量的一致性。
- 新型便携设备对比实验室金标准。
- 两种影像学测量（如射血分数）的互换性评估。

## 11. 相关方法

- [[组内相关系数（Intraclass Correlation Coefficient, ICC）]]
- [[Cohen Kappa一致性系数（Cohen's Kappa）]]
- [[Pearson相关（Pearson Correlation）]]

## 12. 参考资料

- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet*. 1986;327(8476):307-310.
- Bland JM, Altman DG. Measuring agreement in method comparison studies. *Stat Methods Med Res*. 1999;8(2):135-160.
- Giavarina D. Understanding Bland Altman analysis. *Biochem Med*. 2015;25(2):141-151.

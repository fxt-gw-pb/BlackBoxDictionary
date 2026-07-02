---
title: Holm程序
english_name: Holm Procedure
slug: holm-procedure
aliases: [Holm correction, Holm step-down, "Holm程序（Holm Procedure）"]
category: 模型评估与解释
subcategory: 多重比较
tags: [医学统计, 数据科学, 多重检验, FWER]
status: 已建
difficulty: basic
question_type: FWER控制
data_type: [P值列表]
outcome_type: [多重假设检验]
python_packages: [statsmodels]
r_packages: [stats]
---

# Holm程序（Holm Procedure）

## 1. 方法概览

### 1.1 定义

Holm 程序是控制 FWER 的逐步法（step-down procedure），通常比 Bonferroni 更有力。

### 1.2 它主要解决什么问题

- 研究问题：如何在保持 FWER 控制的前提下，比 Bonferroni 更少损失检验力。
- 适用任务：严格控制假阳性，同时尽量减少过度保守。
- 常见医学场景：中等规模的多终点比较或多亚组比较。

### 1.3 直觉理解

Holm 方法会先看最小的 p 值，对最有希望显著的检验先用最严格阈值；一旦前面几步通过，后面的阈值会逐步放宽。

## 2. 数学形式

### 2.1 核心公式

对有序 p 值 $p_{(1)} \le \cdots \le p_{(m)}$：

- 若 $p_{(1)} > \alpha/m$，停止；
- 否则拒绝 $H_{(1)}$，继续检验 $p_{(2)} > \alpha/(m-1)$；
- 依此类推。

调整后 p 值可写为：

$$
p_{(i)}^* = \min\left\{1,\ \max\left[(m-i+1)p_{(i)},\ p_{(i-1)}^*\right]\right\}
$$

### 2.2 参数或统计量含义

- $p_{(i)}$：从小到大排序后的第 $i$ 个 p 值。
- $p_{(i)}^*$：Holm 调整后 p 值。

### 2.3 关键假设

- 不要求检验独立。
- 目标是控制 FWER。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：不适用。
- 因变量类型：不适用。
- 数据结构：一列原始 p 值。
- 是否适合高维数据：可用，但仍偏保守。
- 是否适合缺失较多数据：缺失 p 值需先处理。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：可用于其派生检验结果。

### 3.2 示例表格

| hypothesis | p_raw | p_holm |
| --- | --- | --- |
| H1 | 0.0008 | 0.0080 |
| H2 | 0.0030 | 0.0270 |
| H3 | 0.0080 | 0.0640 |
| H4 | 0.0110 | 0.0770 |
| H5 | 0.0180 | 0.1080 |

### 3.3 输入与产出

#### 输入

- 输入数据：原始 p 值。
- 关键变量：排序后的 p 值和目标 $\alpha$。
- 需要预处理的内容：定义检验家族。

#### 产出

- 模型对象/统计结果：Holm 调整后 p 值。
- 参数估计：无。
- 预测结果：无。
- 不确定性指标：体现在 FWER 控制。

## 4. 适用场景

- 适合：希望比 Bonferroni 略有更高检验力，但仍严格控制 FWER。
- 不适合：极大规模发现型研究。
- 使用前需要特别检查的点：是否真的需要 FWER 而不是 FDR。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
```

### 5.2 R

常用包：

- `stats`

```r
p.adjust(pvals, method = "holm")
```

## 6. 结果如何解释

- 核心结果看什么：哪些检验在 Holm 校正后仍显著。
- 每个主要参数如何解释：Holm 调整后 p 值可以直接与 0.05 比较。
- 临床或医学意义如何表达：适合在严格控制假阳性的同时保留更多真实发现。
- 常见误读：Holm 不是 FDR 方法，仍属于 FWER 控制。

## 7. 推荐可视化

- 原始 p 值、Bonferroni 与 Holm 对比图。
- 排序 p 值与 step-down 阈值图。

### 7.1 图像示例

下图中的蓝色折线展示了 Holm 调整后的 p 值，通常比 Bonferroni 略低，因此检验力更高。

![](../../04_示例图像/multiple_testing_adjustments.png)

## 8. 优势、局限与常见坑

### 优势

- 控制 FWER。
- 比 Bonferroni 更有力。
- 对依赖结构要求较少。

### 局限

- 仍然偏保守。
- 多检验场景下发现率可能不高。

### 常见坑

- 和 Hochberg、Hommel 混淆。
- 在高维组学场景下误当默认方法。

## 9. 与相近方法的区别

- 和 Bonferroni 的区别：Holm 是 sequential step-down，更不保守。
- 和 BH 的区别：BH 控制 FDR，不是 FWER。

## 10. 医学研究中的典型应用

- 多终点临床指标的严格校正。
- 多个预设亚组比较。

## 11. 相关方法

- [[Bonferroni校正（Bonferroni Correction）]]
- [[Benjamini-Hochberg程序（Benjamini-Hochberg Procedure）]]

## 12. 参考资料

- Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat*. 1979;6(2):65-70.
- R Core Team. `p.adjust`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) （访问日期：2026-07-02）

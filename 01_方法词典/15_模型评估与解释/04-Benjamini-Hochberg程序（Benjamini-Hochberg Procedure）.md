---
title: Benjamini-Hochberg程序
english_name: Benjamini-Hochberg Procedure
aliases: [BH procedure, FDR control, "Benjamini-Hochberg程序（Benjamini-Hochberg Procedure）"]
category: 模型评估与解释
subcategory: 多重比较
tags: [医学统计, 数据科学, 多重检验, FDR]
status: draft
difficulty: basic
question_type: FDR控制
data_type: [P值列表]
outcome_type: [多重假设检验]
python_packages: [statsmodels]
r_packages: [stats]
related_methods: [Bonferroni校正, Holm程序, 多重检验与错误率控制]
---

# Benjamini-Hochberg程序（Benjamini-Hochberg Procedure）

## 1. 方法概览

### 1.1 定义

Benjamini-Hochberg（BH）程序是控制 FDR 的经典方法，特别适合高维发现型研究。

### 1.2 它主要解决什么问题

- 研究问题：当允许少量假阳性时，如何在控制错误发现比例的同时尽量保留发现能力。
- 适用任务：大规模并行检验、发现型研究、多组学筛选。
- 常见医学场景：基因组学、转录组学、蛋白组学、微生物组差异分析。

### 1.3 直觉理解

BH 不要求“一个假阳性都不能有”，而是允许在所有被宣布显著的结果里，保有一个可控的假发现比例，因此通常比 FWER 方法更有力。

## 2. 数学形式

### 2.1 核心公式

对有序 p 值 $p_{(1)} \le \cdots \le p_{(m)}$，找到最大的 $k$ 满足：

$$
p_{(k)} \le \frac{k}{m}\alpha
$$

然后拒绝 $H_{(1)}, \ldots, H_{(k)}$。

### 2.2 参数或统计量含义

- $m$：总检验数。
- $k$：最后一个通过 BH 阈值的排序位置。
- FDR：错误发现率，关注“显著结果中错的比例”。

### 2.3 关键假设

- 经典 BH 保证通常在独立或正相关（PRDS）检验下成立。
- 目标是控制 FDR，而不是 FWER。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：不适用。
- 因变量类型：不适用。
- 数据结构：一列原始 p 值。
- 是否适合高维数据：非常适合。
- 是否适合缺失较多数据：缺失 p 值需先处理。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：可用于其派生检验结果。

### 3.2 示例表格

| hypothesis | p_raw | p_bh |
| --- | --- | --- |
| H1 | 0.0008 | 0.0080 |
| H2 | 0.0030 | 0.0150 |
| H3 | 0.0080 | 0.0267 |
| H4 | 0.0110 | 0.0275 |
| H5 | 0.0180 | 0.0360 |
| H6 | 0.0290 | 0.0483 |

### 3.3 输入与产出

#### 输入

- 输入数据：原始 p 值。
- 关键变量：排序后的 p 值、目标 FDR 水平。
- 需要预处理的内容：定义同一检验家族。

#### 产出

- 模型对象/统计结果：BH 调整后 p 值，或显著结果集合。
- 参数估计：无。
- 预测结果：无。
- 不确定性指标：体现在 FDR 控制目标。

## 4. 适用场景

- 适合：大规模发现型研究。
- 不适合：不能容忍任何假阳性的高风险决策场景。
- 使用前需要特别检查的点：依赖结构、是否真正关心 FDR 而不是 FWER。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
```

### 5.2 R

常用包：

- `stats`

```r
p.adjust(pvals, method = "BH")
```

## 6. 结果如何解释

- 核心结果看什么：在 FDR 控制下仍显著的检验数。
- 每个主要参数如何解释：BH 调整后 p 值越小，说明该发现越稳健。
- 临床或医学意义如何表达：适合在高通量研究里平衡发现率和错误率。
- 常见误读：BH 控制的是“平均错误发现比例”，不是“绝不出错”。

## 7. 推荐可视化

- 原始 p 值与 BH 调整后 p 值对比图。
- 排序 p 值与 BH 阈值线。
- volcano / Manhattan plot 叠加 FDR 阈值。

### 7.1 图像示例

下图中的绿色折线展示了 BH 调整后的 p 值，相比 FWER 方法通常更保留显著信号。

![](../../04_示例图像/multiple_testing_adjustments.png)

## 8. 优势、局限与常见坑

### 优势

- 比 FWER 方法更有力。
- 特别适合高维筛选场景。
- 结果通常更符合发现型研究需求。

### 局限

- 允许一定比例假阳性。
- 对依赖结构有条件要求。
- 在极端小样本场景中不一定理想。

### 常见坑

- 把 FDR 误解为“单个结果出错概率”。
- 在不能容忍假阳性的场景下仍使用 BH。
- 忽略检验家族界定。

## 9. 与相近方法的区别

- 和 Bonferroni / Holm 的区别：BH 控制 FDR，而不是 FWER。
- 和 BY 的区别：BY 更保守，可处理更一般的依赖结构。

## 10. 医学研究中的典型应用

- GWAS 和多组学发现型分析。
- 多个分子标志物筛选。
- 微生物组差异丰度比较。

## 11. 相关方法

- [[多重检验与错误率控制（Multiple Testing and Error Rate Control）]]
- [[Bonferroni校正（Bonferroni Correction）]]
- [[Holm程序（Holm Procedure）]]

## 12. 参考资料

- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B*. 1995;57(1):289-300.
- R Core Team. `p.adjust`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) （访问日期：2026-07-02）

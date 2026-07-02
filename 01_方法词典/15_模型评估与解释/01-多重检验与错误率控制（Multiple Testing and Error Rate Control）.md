---
title: 多重检验与错误率控制
english_name: Multiple Testing and Error Rate Control
slug: multiple-testing-and-error-rate-control
aliases: [multiple testing, FWER, FDR, "多重检验与错误率控制（Multiple Testing and Error Rate Control）"]
category: 模型评估与解释
subcategory: 多重比较
tags: [医学统计, 数据科学, 多重检验, FWER, FDR]
status: 已建
difficulty: intermediate
question_type: 假阳性控制
data_type: [P值列表, 多重检验结果]
outcome_type: [多重假设检验]
python_packages: [statsmodels]
r_packages: [stats]
---

# 多重检验与错误率控制（Multiple Testing and Error Rate Control）

## 1. 方法概览

### 1.1 定义

多重检验与错误率控制关注的是：当同时进行很多次假设检验时，如何控制整体假阳性风险或错误发现比例。

### 1.2 它主要解决什么问题

- 研究问题：一次做很多个检验时，如何避免仅靠运气就“发现”大量假阳性。
- 适用任务：高通量分析、多指标比较、亚组比较、多特征筛选。
- 常见医学场景：基因组学、蛋白组学、微生物组、多终点临床研究。

### 1.3 直觉理解

当检验次数很多时，即使每次单独控制在 0.05，整体至少出现一次假阳性的概率也会迅速升高。多重检验方法就是在“发现更多信号”和“控制错误发现”之间做取舍。

## 2. 数学形式

### 2.1 核心公式

常见两类目标：

$$
\mathrm{FWER}=P(V \ge 1)
$$

$$
\mathrm{FDR}=E\left(\frac{V}{R \vee 1}\right)
$$

其中 $V$ 是错误拒绝的真零假设个数，$R$ 是总拒绝数。

### 2.2 参数或统计量含义

- FWER：family-wise error rate，强调“至少出现一次假阳性”的风险。
- FDR：false discovery rate，强调“被宣称显著的发现中有多少比例可能是假的”。

### 2.3 关键假设

- 输入通常是一组同一研究问题下的 p 值。
- 不同方法对 p 值之间的独立性 / 正相关性要求不同。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：不适用，核心输入是多重检验产生的一组 p 值。
- 因变量类型：不适用。
- 数据结构：一列原始 p 值，外加方法对应的调整后 p 值或阈值。
- 是否适合高维数据：非常适合。
- 是否适合缺失较多数据：缺失的 p 值需在进入校正前明确处理。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：可用于重复测量后衍生出的多重检验结果。

### 3.2 示例表格

下面是一组排序后的原始 p 值及不同校正方法的调整结果：

| hypothesis | p_raw | p_bonf | p_holm | p_bh |
| --- | --- | --- | --- | --- |
| H1 | 0.0008 | 0.0080 | 0.0080 | 0.0080 |
| H2 | 0.0030 | 0.0300 | 0.0270 | 0.0150 |
| H3 | 0.0080 | 0.0800 | 0.0640 | 0.0267 |
| H4 | 0.0110 | 0.1100 | 0.0770 | 0.0275 |
| H5 | 0.0180 | 0.1800 | 0.1080 | 0.0360 |

### 3.3 输入与产出

#### 输入

- 输入数据：一组 p 值。
- 关键变量：原始 p 值、目标错误率（如 0.05）、所选校正方法。
- 需要预处理的内容：确认这些 p 值属于同一多重检验家族。

#### 产出

- 模型对象/统计结果：调整后 p 值或校正阈值。
- 参数估计：无。
- 预测结果：无。
- 不确定性指标：主要体现为错误率控制目标本身。

## 4. 适用场景

- 适合：一次做很多个检验的任何场景。
- 不适合：只有一个或极少数预先指定检验时。
- 使用前需要特别检查的点：检验家族定义、控制目标是 FWER 还是 FDR。

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
p.adjust(pvals, method = "bonferroni")
p.adjust(pvals, method = "holm")
p.adjust(pvals, method = "BH")
```

## 6. 结果如何解释

- 核心结果看什么：哪些原始显著结果在校正后仍然显著。
- 每个主要参数如何解释：校正后的 p 值可与目标水平直接比较。
- 临床或医学意义如何表达：应说明控制的是 FWER 还是 FDR。
- 常见误读：未校正显著并不等于校正后仍然显著。

## 7. 推荐可视化

- 原始 p 值与调整后 p 值对比图。
- 不同校正方法的阈值比较图。
- Manhattan / volcano plot 上叠加校正阈值。

### 7.1 图像示例

下图比较原始 p 值与 Bonferroni、Holm、BH 三种方法的调整结果，适合作为多重检验控制思路的总览图。

![](../../04_示例图像/multiple_testing_adjustments.png)

## 8. 优势、局限与常见坑

### 优势

- 能系统控制大规模检验的假阳性。
- 有清晰的错误率目标。
- 适用于现代高维数据分析。

### 局限

- 更严格的方法会降低发现率。
- 不同方法对依赖结构要求不同。
- “检验家族”的定义并不总是唯一。

### 常见坑

- 只报未校正 p 值。
- 混淆 FWER 和 FDR。
- 把不相关的一堆检验强行合并或错误拆分家族。

## 9. 与相近方法的区别

- FWER 控制更严格，适合不能容忍任何假阳性的情形。
- FDR 控制更宽松，适合高通量发现型研究。
- Bonferroni / Holm / BH 分别是不同控制目标或控制强度下的代表方法。

## 10. 医学研究中的典型应用

- GWAS 和多组学高通量分析。
- 多终点临床研究。
- 大量亚组和特征筛选检验。

## 11. 相关方法

- [[Bonferroni校正（Bonferroni Correction）]]
- [[Holm程序（Holm Procedure）]]
- [[Benjamini-Hochberg程序（Benjamini-Hochberg Procedure）]]

## 12. 参考资料

- Shaffer JP. Multiple hypothesis testing. *Annu Rev Psychol*. 1995;46:561-584.
- R Core Team. `p.adjust`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) （访问日期：2026-07-02）

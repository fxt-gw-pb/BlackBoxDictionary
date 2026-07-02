---
title: 组内相关系数
english_name: Intraclass Correlation Coefficient, ICC
slug: intraclass-correlation-coefficient-icc
aliases: [ICC, 组内相关, 内部相关系数, "组内相关系数（Intraclass Correlation Coefficient, ICC）"]
category: 描述统计与统计推断
subcategory: 信度分析
tags: [医学统计, 数据科学, 信度, 重复测量, 方差分解]
status: 已建
difficulty: intermediate
question_type: 连续测量的评分者/重测信度
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [pingouin]
r_packages: [irr, psych]
---

# 组内相关系数（Intraclass Correlation Coefficient, ICC）

## 1. 方法概览

### 1.1 定义

组内相关系数衡量连续测量的信度，即同一对象被不同评分者或不同次测量时结果的一致程度，本质是「对象间方差占总方差的比例」。

### 1.2 它主要解决什么问题

- 研究问题：一项连续测量在不同评分者或不同时间点下有多可重复。
- 适用任务：评分者间信度、重测信度、仪器一致性、量表信度。
- 常见医学场景：血压/超声测量的可重复性、量表总分的稳定性、多阅片者定量读数。

### 1.3 直觉理解

若测量可靠，同一个人重复测得的值应彼此接近，而不同人之间应有真实差异。ICC 就是把「真实的个体间差异」从「测量噪声」中分离出来，占比越高说明测量越可靠。

## 2. 数学形式

### 2.1 核心公式

以单向随机模型为例：

$$
\mathrm{ICC}=\frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2}
$$

双向模型中还可区分「绝对一致」与「一致性」，绝对一致额外计入评分者系统偏差 $\sigma_r^2$：

$$
\mathrm{ICC}_{\text{agreement}}=\frac{\sigma_b^2}{\sigma_b^2+\sigma_r^2+\sigma_e^2}
$$

### 2.2 参数或统计量含义

- $\sigma_b^2$：对象（被测个体）间方差，即「真实」信号。
- $\sigma_e^2$：残差方差，即测量误差。
- $\sigma_r^2$：评分者主效应方差（系统偏差）。
- Shrout-Fleiss 记号 ICC(1,1)/(2,1)/(3,1) 区分模型与单次/均值形式。

### 2.3 关键假设

- 方差分量模型（单向或双向 ANOVA）设定正确。
- 明确区分「一致性 consistency」与「绝对一致 agreement」目标。
- 明确报告单次评分还是 $k$ 次均值的信度。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：无，多列（评分者/时间点）连续测量。
- 因变量类型：连续测量值。
- 数据结构：对象 × 评分者的宽表或长表。
- 是否适合高维数据：可多评分者。
- 是否适合缺失较多数据：混合模型可处理部分缺失。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：正是其用途（重测信度）。

### 3.2 示例表格

3 名评分者对 5 名受试者的评分：

| Subject | Rater1 | Rater2 | Rater3 |
| --- | --- | --- | --- |
| 1 | 9 | 8 | 9 |
| 2 | 6 | 6 | 5 |
| 3 | 8 | 7 | 8 |
| 4 | 3 | 4 | 3 |
| 5 | 7 | 8 | 7 |

### 3.3 输入与产出

#### 输入

- 输入数据：对象在多个评分者/时点上的测量。
- 关键变量：模型类型（1/2/3）、目标（一致性/绝对一致）、单次/均值。
- 需要预处理的内容：整理成对象 × 评分者结构、处理缺失。

#### 产出

- 模型对象/统计结果：ICC 及 95% CI，方差分量。
- 参数估计：各方差分量。
- 预测结果：无。
- 不确定性指标：ICC 置信区间、F 检验。

## 4. 适用场景

- 适合：连续测量的评分者间/重测信度评估。
- 不适合：分类结局（用 [[Cohen Kappa一致性系数（Cohen's Kappa）]]）、评估固定偏差与量程依赖（用 [[Bland-Altman分析（Bland-Altman Analysis）]]）。
- 使用前需要特别检查的点：选对模型（1/2/3）与形式（单次/均值）、一致性 vs 绝对一致。

## 5. 实现

### 5.1 Python

常用包：

- `pingouin`

```python
import pingouin as pg
import pandas as pd

long = pd.DataFrame({
    "subject": [1,1,1,2,2,2],
    "rater":   ["r1","r2","r3","r1","r2","r3"],
    "score":   [9,8,9,6,6,5],
})
icc = pg.intraclass_corr(data=long, targets="subject",
                         raters="rater", ratings="score")
print(icc[["Type","ICC","CI95%"]])
```

### 5.2 R

常用包：

- `irr`

```r
library(irr)
m <- matrix(c(9,8,9, 6,6,5, 8,7,8, 3,4,3, 7,8,7),
            ncol = 3, byrow = TRUE)
icc(m, model = "twoway", type = "agreement", unit = "single")
```

## 6. 结果如何解释

- 核心结果看什么：ICC 值、置信区间下限、所选模型/形式。
- 每个主要参数如何解释：常用 Koo-Li 标准——$<0.5$ 差、$0.5$–$0.75$ 中、$0.75$–$0.9$ 好、$>0.9$ 优。
- 临床或医学意义如何表达：高 ICC 支持该测量可用于个体随访与多中心比较。
- 常见误读：混淆「一致性」与「绝对一致」；只报点估计不报 CI 下限。

## 7. 推荐可视化

- 评分者两两散点图 + 对角线。
- 各对象测量的误差棒图。
- 方差分量堆叠条形图。

## 8. 优势、局限与常见坑

### 优势

- 直接给出 0–1 的信度尺度，易沟通。
- 可分解方差、区分一致性与绝对一致。
- 支持多评分者与均值信度。

### 局限

- 依赖对象间的真实变异（同质样本会低估）。
- 模型选择不当会得到很不同的数值。
- 不显示偏差是否随量程变化（需 Bland-Altman 补充）。

### 常见坑

- 不声明 ICC(1/2/3) 与单次/均值，导致不可复现。
- 在极同质人群上报低 ICC，误判测量不可靠。
- 用 Pearson 相关代替 ICC（前者对系统偏差不敏感）。

## 9. 与相近方法的区别

- 和 [[Pearson相关（Pearson Correlation）]] 的区别：Pearson 只看线性相关、对系统偏差不敏感；ICC 计入偏差与量级一致。
- 和 [[Cohen Kappa一致性系数（Cohen's Kappa）]] 的区别：Kappa 用于分类，ICC 用于连续。
- 和 [[Bland-Altman分析（Bland-Altman Analysis）]] 的区别：Bland-Altman 展示偏差与一致限，ICC 给单一信度指数。

## 10. 医学研究中的典型应用

- 血压计/超声定量测量的重测与阅片者信度。
- 量表总分的评分者间与重测稳定性。
- 影像组学特征在不同扫描仪/分割者间的稳定性。

## 11. 相关方法

- [[Cohen Kappa一致性系数（Cohen's Kappa）]]
- [[Bland-Altman分析（Bland-Altman Analysis）]]
- [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]

## 12. 参考资料

- Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychol Bull*. 1979;86(2):420-428.
- Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *J Chiropr Med*. 2016;15(2):155-163.
- McGraw KO, Wong SP. Forming inferences about some intraclass correlation coefficients. *Psychol Methods*. 1996;1(1):30-46.

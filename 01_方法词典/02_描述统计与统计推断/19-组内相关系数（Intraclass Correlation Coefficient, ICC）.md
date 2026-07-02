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
r_packages: [irr]
---

# 组内相关系数（Intraclass Correlation Coefficient, ICC）

## 1. 方法概览

### 1.1 一句话本质

ICC 是「真实的个体间差异」占「总变异」的比例：若同一个人重复测量都很接近、而不同人之间差异明显，说明测量可靠，ICC 就高。

### 1.2 定义

组内相关系数度量连续测量的信度，即同一对象被不同评分者或不同次测量时结果的一致程度，本质是把总方差分解为「对象间方差」与「测量误差方差」后，前者所占的比例。

### 1.3 它主要解决什么问题

- 研究问题：一项连续测量在不同评分者或不同时间点下有多可重复？
- 适用任务：评分者间信度、重测信度、仪器/量表一致性。
- 常见医学场景：血压/超声测量的可重复性、量表总分的稳定性、多阅片者定量读数、影像组学特征的稳定性。

### 1.4 直觉与类比

想象给同一批人反复量身高。若尺子可靠，同一个人每次量得都差不多（误差小），而人和人之间高矮分明（真实差异大）。ICC 就是把「人与人之间的真实高矮差」从「量尺的抖动」里分出来，看真实差异占了总变异的几成——占比越高，说明这把尺子越能区分不同的人。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

「测量可靠」到底指什么？如果所有人身高都差不多（同质人群），再好的尺子也难显出「区分力」；如果尺子噪声很大，真实差异会被淹没。根本困难是：**如何把「测量能捕捉的真实个体差异」与「测量本身的噪声」定量地分开？**

### 2.2 关键洞察

用**方差分解**：把每次测量的总变异拆成「对象间方差 $\sigma_b^2$」（不同人真实不同）和「误差方差 $\sigma_e^2$」（同一人重复测的抖动），双向模型还可分出「评分者系统偏差 $\sigma_r^2$」。信度就是「信号占总变异的比例」$\sigma_b^2/(\sigma_b^2+\sigma_e^2)$。这也解释了为何 ICC 依赖人群：同质人群 $\sigma_b^2$ 小、ICC 天然偏低。

### 2.3 与朴素/相邻做法的对比

- 相对 **Pearson 相关**：Pearson 只看两列是否线性相关、**对系统偏差不敏感**（评分者 B 一律比 A 高 5 分，Pearson 仍可为 1）；ICC 计入偏差与量级一致，才是真正的信度。
- 相对 **Cohen's Kappa**：Kappa 用于分类，ICC 用于连续。
- 相对 **Bland-Altman**：Bland-Altman 展示偏差与量程依赖的「图」，ICC 给单一信度「指数」，二者互补。

## 3. 数学形式

### 3.1 核心公式

单向随机模型的信度：

$$
\mathrm{ICC}=\frac{\sigma_b^2}{\sigma_b^2+\sigma_e^2}
$$

由单因素方差分析的均方估计（$k$ 为每对象测量次数）：

$$
\mathrm{ICC}(1,1)=\frac{\mathrm{MS}_b-\mathrm{MS}_e}{\mathrm{MS}_b+(k-1)\mathrm{MS}_e}
$$

这个式子在说：信度 = 对象间方差 ÷ 总方差；用组间均方与组内均方可把方差分量估出来。

### 3.2 推导脉络

- 把测量写成「总均值 + 对象效应 + 误差」的方差分量模型。
- 对象效应方差 $\sigma_b^2$ 是信号，误差方差 $\sigma_e^2$ 是噪声；ICC 是信号占比。
- 用单因素 ANOVA 的 $\mathrm{MS}_b,\mathrm{MS}_e$ 反解方差分量，得上面的估计式。
- 双向模型区分「一致性 consistency」（不计评分者系统偏差）与「绝对一致 agreement」（计入偏差 $\sigma_r^2$）；并区分「单次」与「$k$ 次均值」的信度（Shrout-Fleiss 记号 ICC(1/2/3,1/k)）。

### 3.3 参数与统计量含义

- $\sigma_b^2$：对象间方差（信号）。
- $\sigma_e^2$：误差方差（噪声）；$\sigma_r^2$：评分者偏差方差。
- $k$：每对象测量次数。
- ICC：$[0,1]$，越高越可靠；需注明模型(1/2/3)与形式(单次/均值)。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 方差模型正确 | 单向/双向选对 | ICC 数值偏差 | 依设计选模型 |
| 目标明确 | 一致性 vs 绝对一致 | 报错类型误导 | 按用途选 |
| 人群有代表性 | 有真实个体变异 | 同质人群 ICC 偏低 | 看对象间离散 |

## 4. 手把手算例

对一批对象各测 $k=2$ 次，做单因素 ANOVA 得组间均方 $\mathrm{MS}_b=25$、组内（误差）均方 $\mathrm{MS}_e=1$。

**一步步计算（ICC(1,1)）：**

- 代入：$\mathrm{ICC}=\dfrac{\mathrm{MS}_b-\mathrm{MS}_e}{\mathrm{MS}_b+(k-1)\mathrm{MS}_e}=\dfrac{25-1}{25+(2-1)\times1}=\dfrac{24}{26}\approx0.92$。
- 反解方差分量：$\sigma_e^2=\mathrm{MS}_e=1$，$\sigma_b^2=\dfrac{\mathrm{MS}_b-\mathrm{MS}_e}{k}=\dfrac{24}{2}=12$；校验 $\dfrac{12}{12+1}\approx0.92$ ✓。

**结论：** ICC≈0.92，属「优」信度——对象间的真实差异（方差 12）远大于测量噪声（方差 1），说明这项测量能可靠地区分不同个体、可用于个体随访。若换成很同质的人群（$\mathrm{MS}_b$ 只比 $\mathrm{MS}_e$ 略大，如 2 vs 1），ICC 会骤降到 $\dfrac{1}{3}\approx0.33$——**同一把尺子在不同人群里 ICC 可以差很多**，这提醒解读时必须结合人群变异。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：无，多列（评分者/时间点）连续测量。
- 因变量类型：连续测量值。
- 数据结构：对象 × 评分者的宽表或长表。
- 是否适合高维数据：可多评分者。
- 是否适合缺失较多数据：混合模型可处理部分缺失。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：正是其用途（重测信度）。

### 5.2 示例表格

| Subject | Rater1 | Rater2 |
| --- | --- | --- |
| 1 | 9 | 8 |
| 2 | 6 | 5 |
| 3 | 3 | 4 |

### 5.3 输入与产出

#### 输入

- 输入数据：对象在多个评分者/时点上的测量。
- 关键变量：模型(1/2/3)、目标(一致性/绝对)、单次/均值。
- 需要预处理的内容：整理成对象 × 评分者结构、处理缺失。

#### 产出

- 模型对象/统计结果：ICC 及 95% CI、方差分量。
- 参数估计：各方差分量。
- 预测结果：无。
- 不确定性指标：ICC 置信区间、F 检验。

## 6. 适用场景

- 适合：连续测量的评分者间/重测信度评估。
- 不适合：分类结局（用 Kappa）、评估固定偏差与量程依赖（用 Bland-Altman）。
- 使用前需要特别检查的点：选对模型(1/2/3)与形式(单次/均值)、一致性 vs 绝对一致。

## 7. 实现

### 7.1 Python

常用包：

- `pingouin`

```python
import pingouin as pg
import pandas as pd

long = pd.DataFrame({
    "subject": [1,1,2,2,3,3],
    "rater":   ["r1","r2"]*3,
    "score":   [9,8, 6,5, 3,4],
})
icc = pg.intraclass_corr(data=long, targets="subject",
                         raters="rater", ratings="score")
print(icc[["Type","ICC","CI95%"]])
```

### 7.2 R

常用包：

- `irr`

```r
library(irr)
m <- matrix(c(9,8, 6,5, 3,4), ncol = 2, byrow = TRUE)
icc(m, model = "twoway", type = "agreement", unit = "single")
```

## 8. 结果如何解读

- 核心结果看什么：ICC 值、置信区间下限、所选模型/形式。
- 每个主要参数如何解读：Koo-Li——$\lt0.5$ 差、$0.5$–$0.75$ 中、$0.75$–$0.9$ 好、$\gt0.9$ 优。
- 临床或医学意义如何表达：高 ICC 支持该测量用于个体随访与多中心比较。
- 常见误读：混淆「一致性」与「绝对一致」；只报点估计不报 CI 下限。

## 9. 假设诊断与稳健性

- 模型选择：随机评分者用双向随机、固定评分者用双向混合、单向仅当评分者各不同。
- 人群变异：同质样本会低估 ICC，解读须结合对象间离散。
- 一致性 vs 绝对：关心「排序一致」用 consistency；关心「数值可换」用 agreement。
- 报告规范：注明 ICC(模型,形式) 与 95% CI（Koo-Li 指南）。

## 10. 推荐可视化

- 评分者两两散点图 + 对角线。
- 各对象测量的误差棒图。
- 方差分量堆叠条形图。

## 11. 优势、局限与常见坑

### 优势

- 给 0–1 的信度尺度，易沟通。
- 分解方差、区分一致性与绝对一致。
- 支持多评分者与均值信度。

### 局限

- 依赖对象间真实变异（同质样本低估）。
- 模型选择不当数值差很大。
- 不显示偏差是否随量程变化。

### 常见坑

- 不声明 ICC(1/2/3) 与单次/均值。
- 在极同质人群报低 ICC，误判不可靠。
- 用 Pearson 代替 ICC（前者对系统偏差不敏感）。

## 12. 与相近方法的区别

- 和 **Pearson 相关**：Pearson 只看线性相关、忽略系统偏差；ICC 计入偏差与量级一致。
- 和 **Cohen's Kappa**：Kappa 分类、ICC 连续。
- 和 **Bland-Altman**：Bland-Altman 展示偏差与一致限，ICC 给单一指数。
- 如何选择：连续信度指数 → ICC；方法可换性 → Bland-Altman；分类一致 → Kappa。

## 13. 医学研究中的典型应用

- 血压计/超声定量测量的重测与阅片者信度。
- 量表总分的评分者间与重测稳定性。
- 影像组学特征在不同扫描仪/分割者间的稳定性。

## 14. 关键术语

- **对象间方差（$\sigma_b^2$）**：不同对象真实水平的差异（信号）。
- **误差方差（$\sigma_e^2$）**：同一对象重复测量的抖动（噪声）。
- **一致性 vs 绝对一致**：是否计入评分者系统偏差。
- **Shrout-Fleiss 模型**：ICC(1/2/3, 1/k) 的分类体系。
- **信度（Reliability）**：信号占总变异的比例。

## 15. 相关方法

- [[Cohen Kappa一致性系数（Cohen's Kappa）]]
- [[Bland-Altman分析（Bland-Altman Analysis）]]
- [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]

## 16. 参考资料

- Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychol Bull*. 1979;86(2):420-428.
- Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients. *J Chiropr Med*. 2016;15(2):155-163.
- McGraw KO, Wong SP. Forming inferences about some intraclass correlation coefficients. *Psychol Methods*. 1996;1(1):30-46.

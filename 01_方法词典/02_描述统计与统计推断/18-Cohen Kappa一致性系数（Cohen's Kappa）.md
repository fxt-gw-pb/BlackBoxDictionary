---
title: Cohen's Kappa一致性系数
english_name: Cohen's Kappa
slug: cohen-s-kappa
aliases: [Cohen Kappa, kappa系数, 科恩kappa, "Cohen Kappa一致性系数（Cohen's Kappa）"]
category: 描述统计与统计推断
subcategory: 一致性分析
tags: [医学统计, 数据科学, 一致性, 评分者信度, 分类数据]
status: 已建
difficulty: basic
question_type: 分类评分的评分者间一致性
data_type: [列联表]
outcome_type: [分类变量]
python_packages: [scikit-learn, statsmodels]
r_packages: [irr, vcd]
---

# Cohen Kappa一致性系数（Cohen's Kappa）

## 1. 方法概览

### 1.1 定义

Cohen's Kappa 衡量两名评分者对同一批对象做分类判断时的一致程度，并扣除仅凭机遇就能达成的一致，是评分者间信度的标准指标。

### 1.2 它主要解决什么问题

- 研究问题：两位医生／两种方法对同一批病例的分类判断有多一致。
- 适用任务：评分者间一致性、诊断可重复性、编码信度评估。
- 常见医学场景：影像分级、病理诊断、量表条目编码、疾病是否存在的判读。

### 1.3 直觉理解

两人分类结果的原始一致率会「虚高」，因为即使随便猜也可能碰巧一致。Kappa 把这部分机遇一致剔除后，再看实际一致比机遇一致「多出多少」，因此更能反映真实的判断一致性。

## 2. 数学形式

### 2.1 核心公式

$$
\kappa=\frac{p_o-p_e}{1-p_e}
$$

加权 Kappa（用于有序类别）引入权重 $w_{ij}$：

$$
\kappa_w=1-\frac{\sum_{i,j} w_{ij}\,p_{ij}}{\sum_{i,j} w_{ij}\,p_{i\cdot}p_{\cdot j}}
$$

### 2.2 参数或统计量含义

- $p_o$：观测到的一致比例（对角线之和）。
- $p_e$：期望（机遇）一致比例，由两边缘分布相乘得到。
- $w_{ij}$：类别 $i$ 与 $j$ 不一致的代价，线性或二次权重最常用。
- $\kappa$ 取值范围一般在 $-1$ 到 $1$，$0$ 表示仅达机遇水平。

### 2.3 关键假设

- 两评分者相互独立评分，类别定义相同且互斥穷尽。
- 对象为随机样本；边缘分布会影响 Kappa 的量级。
- 名义类别用无权重 Kappa，有序类别用加权 Kappa。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：无，两列分类评分。
- 因变量类型：名义或有序分类。
- 数据结构：同一批对象两位评分者的配对分类，或 $k\times k$ 列联表。
- 是否适合高维数据：仅二评分者；多评分者用 Fleiss' Kappa。
- 是否适合缺失较多数据：需成对完整评分。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：可用于同一评分者两次评分（重测信度）。

### 3.2 示例表格

两位放射科医师对 100 张片子分「阴性/可疑/阳性」：

| 医师A\医师B | 阴性 | 可疑 | 阳性 |
| --- | --- | --- | --- |
| 阴性 | 40 | 4 | 1 |
| 可疑 | 3 | 18 | 5 |
| 阳性 | 1 | 6 | 22 |

### 3.3 输入与产出

#### 输入

- 输入数据：两列等长的分类标签。
- 关键变量：类别集合、是否有序（决定是否加权）。
- 需要预处理的内容：统一类别编码、剔除缺失对。

#### 产出

- 模型对象/统计结果：$\kappa$ 及其标准误、置信区间。
- 参数估计：$p_o$、$p_e$。
- 预测结果：无。
- 不确定性指标：Kappa 的 95% CI、显著性检验。

## 4. 适用场景

- 适合：名义/有序分类的评分者间或重测一致性评估。
- 不适合：连续测量（用 [[组内相关系数（Intraclass Correlation Coefficient, ICC）]] 或 [[Bland-Altman分析（Bland-Altman Analysis）]]）、评分者多于两位（用 Fleiss）。
- 使用前需要特别检查的点：类别患病率是否极端（Kappa 悖论）、类别是否有序。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
from sklearn.metrics import cohen_kappa_score

a = ["neg","pos","susp","neg","pos"]
b = ["neg","pos","neg","neg","susp"]
print(cohen_kappa_score(a, b))                 # 无权重
print(cohen_kappa_score(a, b, weights="quadratic"))  # 有序加权
```

### 5.2 R

常用包：

- `irr`

```r
library(irr)
ratings <- data.frame(
  A = c("neg","pos","susp","neg","pos"),
  B = c("neg","pos","neg","neg","susp")
)
kappa2(ratings)                       # 无权重
kappa2(ratings, weight = "squared")   # 有序加权
```

## 6. 结果如何解释

- 核心结果看什么：$\kappa$ 值、置信区间及是否显著大于 0。
- 每个主要参数如何解释：常用 Landis-Koch 标准——$<0.2$ 差、$0.2$–$0.4$ 尚可、$0.4$–$0.6$ 中等、$0.6$–$0.8$ 好、$>0.8$ 很好。
- 临床或医学意义如何表达：高 Kappa 说明诊断/分级可重复，可支持多中心一致性。
- 常见误读：把原始一致率当 Kappa；忽视患病率极端时 Kappa 会异常偏低（Kappa 悖论）。

## 7. 推荐可视化

- 混淆矩阵热图（观测一致结构）。
- 一致/不一致条形图。
- 多评分者时的一致性矩阵。

## 8. 优势、局限与常见坑

### 优势

- 扣除机遇一致，比原始一致率更严格。
- 有加权版本处理有序类别的「近似一致」。
- 有成熟的标准误与区间。

### 局限

- 受边缘分布/患病率影响大（Kappa 悖论）。
- 仅两评分者；多评分者需 Fleiss。
- 不同类别数间的 Kappa 不可直接比较。

### 常见坑

- 有序类别用无权重 Kappa，低估一致性。
- 不报告患病率与边缘分布，导致 Kappa 难以解释。
- 用单一阈值机械判定「好/坏」。

## 9. 与相近方法的区别

- 和 [[组内相关系数（Intraclass Correlation Coefficient, ICC）]] 的区别：ICC 用于连续测量的信度，Kappa 用于分类。
- 和 [[Pearson卡方独立性检验（Pearson Chi-Squared Test of Independence）]] 的区别：卡方检验关联性，Kappa 度量一致性（对角线）。
- 和百分比一致率的区别：Kappa 扣除了机遇一致。

## 10. 医学研究中的典型应用

- 两位放射/病理医师诊断分级的一致性。
- 量表条目或病历编码的评分者信度。
- 同一诊断工具的重测可重复性。

## 11. 相关方法

- [[组内相关系数（Intraclass Correlation Coefficient, ICC）]]
- [[Bland-Altman分析（Bland-Altman Analysis）]]
- [[Pearson卡方独立性检验（Pearson Chi-Squared Test of Independence）]]

## 12. 参考资料

- Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas*. 1960;20(1):37-46.
- Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174.
- Sim J, Wright CC. The kappa statistic in reliability studies. *Phys Ther*. 2005;85(3):257-268.

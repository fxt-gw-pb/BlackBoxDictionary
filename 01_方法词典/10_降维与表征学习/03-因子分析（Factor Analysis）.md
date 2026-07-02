---
title: 因子分析
english_name: Factor Analysis
slug: factor-analysis
aliases: [factor analysis, exploratory factor analysis, EFA, "因子分析（Factor Analysis）"]
category: 降维与表征学习
subcategory: 潜变量模型
tags: [医学统计, 数据科学, 降维, 潜变量, 心理测量]
status: 已建
difficulty: intermediate
question_type: 潜变量提取与量表结构分析
data_type: [表格数据, 量表数据, 高维特征矩阵]
outcome_type: [无监督表征, 多种]
python_packages: [factor_analyzer, scikit-learn]
r_packages: [psych, stats]
---

# 因子分析（Factor Analysis）

## 1. 方法概览

### 1.1 定义

因子分析是一类潜变量模型，用少数不可直接观测的公共因子解释多个观测变量之间的相关结构。它常用于问卷量表、心理测量、症状维度和综合指标构建。

### 1.2 它主要解决什么问题

- 研究问题：多个观测指标背后是否存在少数共同潜在维度。
- 适用任务：探索性因子分析、量表结构检验、变量降维、潜在构念命名。
- 常见医学场景：生活质量量表维度分析，症状群识别，心理健康评分结构探索。

### 1.3 直觉理解

如果多个问题或指标总是一起变化，它们可能共同反映同一个潜在概念。因子分析把这种共同变化归结为公共因子，同时把每个变量自身的特殊部分留在误差项中。

## 2. 数学形式

### 2.1 核心公式

设标准化后的观测变量向量为 $x\in\mathbb{R}^{p}$，公共因子为 $f\in\mathbb{R}^{k}$：

$$
x = \Lambda f + \epsilon
$$

其中 $\Lambda$ 为因子载荷矩阵，$\epsilon$ 为特异误差。协方差结构可写为：

$$
\Sigma = \Lambda \Phi \Lambda^\top + \Psi
$$

若假设因子彼此正交，则 $\Phi=I$：

$$
\Sigma = \Lambda \Lambda^\top + \Psi
$$

### 2.2 参数或统计量含义

- $\Lambda$：因子载荷矩阵，表示每个观测变量与公共因子的关系。
- $f$：公共因子或潜变量。
- $\Psi$：特异方差矩阵，表示每个变量未被公共因子解释的部分。
- 共同度：某变量方差中被公共因子解释的比例。
- 因子旋转：通过正交或斜交变换提升载荷解释性。

### 2.3 关键假设

- 观测变量之间的相关性可由少数公共因子解释。
- 特异误差之间通常假设相互独立。
- 样本量足以稳定估计相关矩阵。
- 变量应有足够相关性，完全不相关的变量不适合做因子分析。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、近似连续的量表题项或有序评分。
- 因变量类型：因子分析本身不需要结局变量。
- 数据结构：样本乘以变量/题项矩阵。
- 是否适合高维数据：适合中等维度，变量很多时需注意样本量和稳定性。
- 是否适合缺失较多数据：需先处理缺失，或使用支持缺失机制的估计方法。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：普通因子分析需先明确分析单位，纵向量表可考虑专门模型。

### 3.2 示例表格

以患者报告结局量表为例：

| Pain | Fatigue | Sleep | Anxiety | Depression | SocialFunction |
| --- | --- | --- | --- | --- | --- |
| 4 | 5 | 4 | 2 | 3 | 3 |
| 1 | 2 | 2 | 1 | 1 | 5 |
| 5 | 4 | 5 | 4 | 5 | 2 |
| 2 | 3 | 2 | 2 | 2 | 4 |

### 3.3 输入与产出

#### 输入

- 输入数据：多个相关观测变量或题项。
- 关键变量：提取因子数、估计方法、旋转方式。
- 需要预处理的内容：缺失处理、反向题重编码、变量标准化、相关矩阵适用性检查。

#### 产出

- 模型对象/统计结果：因子载荷矩阵、共同度、特异方差、因子得分。
- 参数估计：$\Lambda$、$\Psi$，斜交旋转时还包括因子相关矩阵。
- 预测结果：不直接预测，可输出因子得分用于后续建模。
- 不确定性指标：可报告载荷置信区间、bootstrap 稳定性或验证性分析结果。

## 4. 适用场景

- 适合：多个指标可能反映少数潜在构念、希望解释变量间相关结构、需要构建综合维度评分的场景。
- 不适合：变量之间相关很弱、目标只是压缩预测特征、需要最大化解释总方差而不关心潜变量含义的场景。
- 使用前需要特别检查的点：KMO、Bartlett 球形检验、因子数选择、旋转后载荷是否可解释。

## 5. 实现

### 5.1 Python

常用包：

- `factor_analyzer`
- `scikit-learn`

```python
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("patient_reported_outcomes.csv")
items = ["Pain", "Fatigue", "Sleep", "Anxiety", "Depression", "SocialFunction"]
X = StandardScaler().fit_transform(df[items])

fa = FactorAnalyzer(n_factors=2, rotation="varimax")
fa.fit(X)

loadings = pd.DataFrame(
    fa.loadings_,
    index=items,
    columns=["Factor1", "Factor2"]
)
scores = fa.transform(X)

print(loadings.round(3))
print(pd.DataFrame(scores, columns=["Factor1", "Factor2"]).head())
```

### 5.2 R

常用包：

- `psych`

```r
library(psych)

items <- df[, c("Pain", "Fatigue", "Sleep", "Anxiety", "Depression", "SocialFunction")]
fit <- fa(items, nfactors = 2, rotate = "varimax", fm = "ml")

print(fit$loadings, cutoff = 0.30)
scores <- factor.scores(items, fit)$scores
head(scores)
```

## 6. 结果如何解释

- 核心结果看什么：载荷矩阵、共同度、因子相关、因子得分分布。
- 每个主要参数如何解释：载荷越大，说明该变量越能代表对应公共因子。
- 临床或医学意义如何表达：因子命名应来自高载荷题项的共同含义，而不是算法自动给出的标签。
- 常见误读：因子数不是越多越好；旋转改变解释方式，但不应被理解为发现了完全新的数据。

## 7. 推荐可视化

- 因子载荷热图。
- 碎石图和平行分析图。
- 因子得分散点图。
- 题项按因子分组的相关矩阵热图。

## 8. 优势、局限与常见坑

### 优势

- 适合解释多个观测指标背后的潜在构念。
- 可将高相关题项压缩为少数可命名维度。
- 在量表开发和医学心理测量中有成熟使用传统。

### 局限

- 因子数和旋转方式存在研究者判断。
- 对样本量、题项质量和相关矩阵稳定性敏感。
- 探索性结果需要外部样本或验证性因子分析确认。

### 常见坑

- 把 PCA 结果直接解释成潜变量。
- 只看特征值大于 1，而不结合平行分析和理论背景。
- 对交叉载荷较高的题项强行命名。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 解释总方差，因子分析解释变量间共同方差并显式建模特异误差。
- 和 [[独立成分分析（Independent Component Analysis, ICA）]] 的区别：ICA 强调成分统计独立，因子分析强调公共因子解释相关结构。
- 和 [[奇异值分解（Singular Value Decomposition, SVD）]] 的区别：SVD 是矩阵代数分解，因子分析是带误差结构的统计潜变量模型。

## 10. 医学研究中的典型应用

- 患者报告结局量表的维度结构探索。
- 症状群或心理健康潜在维度识别。
- 多个生活方式指标合成为可解释的潜在行为因子。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[独立成分分析（Independent Component Analysis, ICA）]]
- [[奇异值分解（Singular Value Decomposition, SVD）]]
- [[线性判别分析（Linear Discriminant Analysis, LDA）]]

## 12. 参考资料

- Fabrigar LR, Wegener DT. *Exploratory Factor Analysis*. Oxford University Press; 2011.
- Harman HH. *Modern Factor Analysis*. 3rd ed. University of Chicago Press; 1976.
- Revelle W. `psych`: Procedures for Psychological, Psychometric, and Personality Research. [https://cran.r-project.org/package=psych](https://cran.r-project.org/package=psych) （访问日期：2026-07-02）

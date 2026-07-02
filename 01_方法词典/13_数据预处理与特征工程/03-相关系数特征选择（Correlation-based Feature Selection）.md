---
title: 相关系数特征选择
english_name: Correlation-based Feature Selection
slug: correlation-based-feature-selection
aliases: [correlation-based feature selection, correlation filter, 相关系数法, "相关系数特征选择（Correlation-based Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 相关分析, 共线性]
status: 已建
difficulty: basic
question_type: 冗余特征过滤
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督预处理, 多种]
python_packages: [pandas, numpy, scikit-learn]
r_packages: [caret, recipes]
---

# 相关系数特征选择（Correlation-based Feature Selection）

## 1. 方法概览

### 1.1 定义

相关系数特征选择是一种基于变量间相关性的过滤方法。它通过计算特征之间，或特征与结局之间的相关系数，删除冗余特征或保留与结局关系更强的特征。

### 1.2 它主要解决什么问题

- 研究问题：如何识别高度相关、信息重复的特征。
- 适用任务：建模前降冗余、缓解共线性、初步特征筛选。
- 常见医学场景：实验室指标共线性处理、影像组学冗余特征删除、临床评分构建前变量整理。

### 1.3 直觉理解

如果两个特征几乎同步变化，它们可能携带类似信息。相关系数法会找出这些“重复表达”的变量，通常只保留其中一个，以降低模型复杂度。

## 2. 数学形式

### 2.1 核心公式

Pearson 相关系数定义为：

$$
r(X,Y)=
\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar x)^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar y)^2}}
$$

设特征相关阈值为 $\theta$，当两个特征满足：

$$
|r(X_j,X_k)|\geq \theta
$$

则认为它们高度相关，可考虑删除其中一个。

### 2.2 参数或统计量含义

- $r$：相关系数，范围为 $[-1,1]$。
- $\theta$：高相关阈值，常见取值为 0.8、0.9 或 0.95。
- Pearson 相关：衡量线性关系。
- Spearman 相关：衡量单调关系，更适合偏态或等级变量。

### 2.3 关键假设

- Pearson 相关主要反映线性关系。
- 高相关特征存在冗余，但不代表其中一个一定无医学意义。
- 删除哪个特征应结合缺失率、测量成本、可解释性和与结局的关系。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、等级变量、部分编码变量。
- 因变量类型：可不使用结局，也可用于特征-结局相关筛选。
- 数据结构：宽表数据。
- 是否适合高维数据：适合，但相关矩阵计算和可视化需注意规模。
- 是否适合缺失较多数据：需明确相关系数的缺失处理方式。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：普通相关矩阵不处理组内相关。

### 3.2 示例表格

以肾功能相关指标预处理为例：

| Creatinine | eGFR | BUN | CystatinC | Age | CKD |
| --- | --- | --- | --- | --- | --- |
| 1.8 | 42 | 29 | 1.6 | 71 | 1 |
| 0.8 | 98 | 12 | 0.7 | 45 | 0 |
| 1.4 | 58 | 24 | 1.2 | 63 | 1 |
| 0.9 | 91 | 14 | 0.8 | 52 | 0 |
| 2.2 | 31 | 35 | 1.9 | 77 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：候选特征矩阵，可选目标变量。
- 关键变量：相关系数类型、阈值、删除规则。
- 需要预处理的内容：缺失处理、异常值检查、变量变换、必要时标准化。

#### 产出

- 模型对象/统计结果：相关矩阵、高相关特征对、保留特征集合。
- 参数估计：无模型参数。
- 预测结果：无。
- 不确定性指标：相关系数置信区间或 P 值可作为辅助。

## 4. 适用场景

- 适合：去除高度冗余变量、缓解共线性、提高模型训练效率。
- 不适合：非线性依赖明显、变量间关系复杂、需要模型自动处理交互的场景。
- 使用前需要特别检查的点：阈值是否过于激进；删除规则是否可解释；是否误删临床关键变量。

## 5. 实现

### 5.1 Python

常用包：

- `pandas`
- `numpy`

```python
import numpy as np
import pandas as pd

df = pd.read_csv("kidney_features.csv")
X = df[["Creatinine", "eGFR", "BUN", "CystatinC", "Age"]]

corr = X.corr(method="pearson").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

threshold = 0.90
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
X_selected = X.drop(columns=to_drop)

print("Dropped:", to_drop)
```

### 5.2 R

常用包：

- `caret`

```r
library(caret)

x <- df[, c("Creatinine", "eGFR", "BUN", "CystatinC", "Age")]
corr <- cor(x, use = "pairwise.complete.obs")
drop_idx <- findCorrelation(corr, cutoff = 0.90)

x_selected <- x[, -drop_idx]
```

## 6. 结果如何解释

- 核心结果看什么：高相关特征对、被删除变量、保留变量的临床合理性。
- 每个主要参数如何解释：阈值越低，删除越多；阈值越高，只删除极强冗余。
- 临床或医学意义如何表达：相关系数法用于减少重复信息，不是判断变量是否有因果作用。
- 常见误读：两个变量高度相关，不代表它们可在所有研究问题中互相替代。

## 7. 推荐可视化

- 相关矩阵热图。
- 高相关网络图。
- 删除前后模型性能对比图。

## 8. 优势、局限与常见坑

### 优势

- 简单直观。
- 能快速识别冗余特征。
- 有助于缓解线性模型共线性。

### 局限

- 主要捕捉线性或单调关系。
- 不直接评估预测贡献。
- 删除规则带有主观性。

### 常见坑

- 只按相关性删变量，不看临床可解释性和测量质量。
- 把 Pearson 相关用于明显非线性关系。
- 在训练测试拆分外统一计算相关矩阵，导致预处理流程不严谨。

## 9. 与相近方法的区别

- 和 [[Spearman秩相关（Spearman Rank Correlation）]] 的区别：Spearman 是一种相关系数；本卡强调用相关矩阵做特征筛选流程。
- 和 [[Ridge回归（Ridge Regression）]] 的区别：Ridge 保留相关变量并收缩系数；相关系数法直接删除部分冗余变量。
- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 把相关特征组合成新成分；相关系数法仍保留原始变量。

## 10. 医学研究中的典型应用

- 从高度相关的实验室指标中保留更稳定或更常用的指标。
- 影像组学建模前删除成对相关过高的纹理特征。
- 临床预测模型中减少共线变量。

## 11. 相关方法

- [[Spearman秩相关（Spearman Rank Correlation）]]
- [[Ridge回归（Ridge Regression）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[方差阈值法（Variance Threshold）]]

## 12. 参考资料

- Guyon I, Elisseeff A. An introduction to variable and feature selection. *J Mach Learn Res*. 2003;3:1157-1182.
- Kuhn M, Johnson K. *Feature Engineering and Selection: A Practical Approach for Predictive Models*. Chapman and Hall/CRC; 2019.
- Kuhn M. Building predictive models in R using the caret package. *J Stat Softw*. 2008;28(5):1-26.

---
title: 嵌入式特征选择
english_name: Embedded Feature Selection
slug: embedded-feature-selection
aliases: [embedded feature selection, embedded method, 嵌入法, "嵌入式特征选择（Embedded Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 嵌入式特征选择
tags: [医学统计, 数据科学, 特征选择, 正则化, 模型选择]
status: 已建
difficulty: intermediate
question_type: 模型训练中的自动特征选择
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [glmnet, caret]
---

# 嵌入式特征选择（Embedded Feature Selection）

## 1. 方法概览

### 1.1 定义

嵌入式特征选择是在模型训练过程中同步完成特征评价或筛选的方法。常见形式包括 L1 正则化模型、树模型特征重要性和基于模型系数的 `SelectFromModel`。

### 1.2 它主要解决什么问题

- 研究问题：如何让模型在学习预测规律的同时自动判断哪些特征重要。
- 适用任务：分类、回归、高维特征筛选、预测建模。
- 常见医学场景：Lasso 影像组学特征筛选、树模型变量重要性筛选、临床风险模型候选变量压缩。

### 1.3 直觉理解

过滤法是建模前先筛一遍特征，嵌入式方法则是“边建模边筛选”。它更贴近最终模型目标，但也更依赖所选模型本身的偏好。

## 2. 数学形式

### 2.1 核心公式

以 L1 正则化为例，嵌入式选择通过稀疏化系数实现：

$$
\hat\beta
=
\arg\min_{\beta}
\left[
L(y,X\beta)+\lambda\sum_{j=1}^{p}|\beta_j|
\right]
$$

选择规则可写为：

$$
\text{keep}(X_j)=I(\hat\beta_j\neq 0)
$$

对基于模型重要性的选择，若模型给出重要性 $I_j$，阈值为 $\theta$：

$$
\text{keep}(X_j)=I(I_j\geq \theta)
$$

### 2.2 参数或统计量含义

- $\lambda$：正则化强度。
- $\hat\beta_j$：模型估计的第 $j$ 个特征系数。
- $I_j$：模型给出的特征重要性。
- 阈值 $\theta$：保留特征的重要性或系数阈值。
- 基模型：用于选择特征的模型，如 Lasso、随机森林、梯度提升树。

### 2.3 关键假设

- 所选模型适合当前任务和数据结构。
- 模型的系数或重要性可作为特征价值的代理。
- 选择过程必须嵌入交叉验证或训练流程，避免信息泄露。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、分类编码变量、高维特征。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：监督学习特征矩阵。
- 是否适合高维数据：适合，尤其是 L1/Elastic Net。
- 是否适合缺失较多数据：需按基模型要求处理。
- 是否适合删失数据：需使用支持删失结局的嵌入式模型。
- 是否适合重复测量数据：需使用分组验证或相应纵向模型。

### 3.2 示例表格

以影像组学特征筛选为例：

| Age | Radiomics_1 | Radiomics_2 | Radiomics_3 | Radiomics_4 | Recurrence |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.28 | 1.42 | -0.12 | 0.88 | 1 |
| 52 | -0.31 | 0.40 | 0.05 | -0.22 | 0 |
| 70 | 0.76 | 1.91 | -0.41 | 1.12 | 1 |
| 47 | -0.22 | 0.15 | 0.11 | -0.34 | 0 |
| 61 | 0.33 | 0.98 | -0.18 | 0.55 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：基模型、正则化强度或重要性阈值、交叉验证方案。
- 需要预处理的内容：缺失处理、标准化、类别编码、训练测试拆分。

#### 产出

- 模型对象/统计结果：选择器、基模型、特征系数或重要性。
- 参数估计：模型系数、重要性得分或正则化参数。
- 预测结果：若直接使用该模型，也可输出预测。
- 不确定性指标：交叉验证性能、特征选择稳定性、测试集表现。

## 4. 适用场景

- 适合：希望特征选择服务于最终预测模型的场景。
- 不适合：只想做模型无关初筛，或需要与具体模型无关的变量解释。
- 使用前需要特别检查的点：是否标准化、基模型是否稳定、选择结果是否随重采样变化很大。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("radiomics_recurrence.csv")
X = df[["Age", "Radiomics_1", "Radiomics_2", "Radiomics_3", "Radiomics_4"]]
y = df["Recurrence"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lasso_logit = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=0.2,
    random_state=42
)

selector = make_pipeline(
    StandardScaler(),
    SelectFromModel(lasso_logit)
)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)
```

### 5.2 R

常用包：

- `glmnet`

```r
library(glmnet)

x <- model.matrix(Recurrence ~ Age + Radiomics_1 + Radiomics_2 + Radiomics_3 + Radiomics_4 - 1, data = df_train)
y <- df_train$Recurrence

fit <- cv.glmnet(x, y, family = "binomial", alpha = 1)
coef_fit <- coef(fit, s = "lambda.min")
selected <- rownames(coef_fit)[as.vector(coef_fit != 0)]
selected <- setdiff(selected, "(Intercept)")
```

## 6. 结果如何解释

- 核心结果看什么：被保留特征、模型性能、选择稳定性。
- 每个主要参数如何解释：正则化越强或阈值越高，保留特征越少。
- 临床或医学意义如何表达：保留特征是当前模型认为有预测贡献的变量，不等于独立危险因素或因果变量。
- 常见误读：把模型重要性当作无条件通用的重要性。

## 7. 推荐可视化

- 正则化路径图。
- 特征重要性或系数条形图。
- 重采样下特征入选频率图。

## 8. 优势、局限与常见坑

### 优势

- 与模型目标一致。
- 可利用正则化控制复杂度。
- 常比纯过滤法更贴近最终预测性能。

### 局限

- 依赖基模型选择。
- 选择结果可能不稳定。
- 解释性受模型偏倚影响。

### 常见坑

- 在交叉验证外先做嵌入式选择。
- 用 Lasso 但不标准化特征。
- 把树模型 impurity importance 当作完全无偏的重要性。

## 9. 与相近方法的区别

- 和 [[单变量特征选择（Univariate Feature Selection）]] 的区别：单变量方法先独立打分；嵌入式方法在模型训练中打分或稀疏化。
- 和 [[Lasso回归（Lasso Regression）]] 的区别：Lasso 是嵌入式特征选择的典型实现之一。
- 和 [[基于树模型的特征选择（Tree-based Feature Selection）]] 的区别：树模型选择是嵌入式方法中的一个重要分支。

## 10. 医学研究中的典型应用

- Lasso 选择影像组学或组学特征。
- 随机森林筛选非线性预测变量。
- Elastic Net 在高相关生物标志物中做稳定筛选。

## 11. 相关方法

- [[Lasso回归（Lasso Regression）]]
- [[Ridge回归（Ridge Regression）]]
- [[弹性网络回归（Elastic Net Regression）]]
- [[基于树模型的特征选择（Tree-based Feature Selection）]]

## 12. 参考资料

- Guyon I, Elisseeff A. An introduction to variable and feature selection. *J Mach Learn Res*. 2003;3:1157-1182.
- Tibshirani R. Regression shrinkage and selection via the lasso. *J R Stat Soc Series B*. 1996;58(1):267-288.
- scikit-learn Developers. `SelectFromModel`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html) （访问日期：2026-07-02）

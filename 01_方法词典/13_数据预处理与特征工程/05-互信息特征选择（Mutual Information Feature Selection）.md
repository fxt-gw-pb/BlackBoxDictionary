---
title: 互信息特征选择
english_name: Mutual Information Feature Selection
slug: mutual-information-feature-selection
aliases: [mutual information feature selection, mutual information, MI, 互信息法, "互信息特征选择（Mutual Information Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 信息论, 非线性关系]
status: 已建
difficulty: intermediate
question_type: 非线性单变量依赖筛选
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [FSelectorRcpp, infotheo]
---

# 互信息特征选择（Mutual Information Feature Selection）

## 1. 方法概览

### 1.1 定义

互信息特征选择用互信息衡量每个特征与目标变量之间共享的信息量，并据此筛选特征。它可以捕捉非线性依赖，不局限于线性相关。

### 1.2 它主要解决什么问题

- 研究问题：哪些特征与结局之间存在较强的信息依赖。
- 适用任务：分类或回归任务中的过滤式特征筛选。
- 常见医学场景：非线性生物标志物筛选、影像组学特征排序、临床表格数据初筛。

### 1.3 直觉理解

互信息衡量“知道这个特征以后，关于结局能少多少不确定”。如果一个特征与结局共享的信息多，它就更可能帮助预测。

## 2. 数学形式

### 2.1 核心公式

互信息定义为：

$$
I(X;Y)=\sum_x\sum_y p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

也可写成熵的形式：

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

或：

$$
I(X;Y)=H(Y)-H(Y\mid X)
$$

特征选择时，常按每个特征 $X_j$ 与目标 $Y$ 的互信息得分排序：

$$
\mathrm{score}(X_j)=I(X_j;Y)
$$

### 2.2 参数或统计量含义

- $I(X;Y)$：两个变量共享的信息量。
- $H(Y)$：目标变量的不确定性。
- $H(Y\mid X)$：已知特征后的剩余不确定性。
- `n_neighbors`：连续变量互信息估计中常见的近邻参数。
- `discrete_features`：指定哪些特征按离散变量处理。

### 2.3 关键假设

- 有足够样本估计变量分布或局部密度。
- 每个特征单独评分，不自动处理特征冗余。
- 连续变量的互信息估计会受样本量、噪声和参数影响。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、离散变量、编码后的分类变量。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：监督学习特征矩阵。
- 是否适合高维数据：适合初筛，但需注意估计稳定性和多重比较。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：普通实现不直接适合删失结局。
- 是否适合重复测量数据：需按独立分析单位处理。

### 3.2 示例表格

以疾病风险预测为例：

| Age | BMI | CRP | ExerciseScore | DietScore | Disease |
| --- | --- | --- | --- | --- | --- |
| 71 | 29.4 | 4.1 | 1 | 2 | 1 |
| 52 | 23.8 | 1.2 | 4 | 4 | 0 |
| 63 | 31.2 | 2.8 | 2 | 3 | 1 |
| 44 | 22.1 | 0.9 | 5 | 5 | 0 |
| 68 | 28.7 | 3.6 | 2 | 2 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：互信息估计方法、近邻数、选择数量或阈值。
- 需要预处理的内容：缺失处理、类别编码、连续/离散变量标记、训练测试划分。

#### 产出

- 模型对象/统计结果：每个特征的互信息得分。
- 参数估计：无传统模型参数。
- 预测结果：无，输出筛选后特征矩阵。
- 不确定性指标：可用 bootstrap 或交叉验证评估得分稳定性。

## 4. 适用场景

- 适合：非线性关系可能存在、特征数量较多、需要模型无关初筛的场景。
- 不适合：样本量很小、连续变量分布估计不稳、强交互但单变量依赖弱的场景。
- 使用前需要特别检查的点：互信息估计是否稳定、是否存在冗余特征、是否在训练集内完成筛选。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split

df = pd.read_csv("disease_features.csv")
X = df[["Age", "BMI", "CRP", "ExerciseScore", "DietScore"]]
y = df["Disease"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

selector = SelectKBest(
    score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
    k=3
)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
print(scores)
```

### 5.2 R

常用包：

- `FSelectorRcpp`

```r
library(FSelectorRcpp)

scores <- information_gain(
  Disease ~ Age + BMI + CRP + ExerciseScore + DietScore,
  data = df_train
)

scores[order(scores$importance, decreasing = TRUE), ]
```

## 6. 结果如何解释

- 核心结果看什么：互信息得分排序、筛选后模型性能、得分稳定性。
- 每个主要参数如何解释：得分越高，特征与结局共享的信息越多；近邻数会影响连续变量估计平滑程度。
- 临床或医学意义如何表达：可说特征与结局有较强统计依赖，但不能说是独立效应或因果效应。
- 常见误读：互信息高不代表模型一定会使用该特征，也不代表该变量不可替代。

## 7. 推荐可视化

- 互信息得分条形图。
- Top-K 特征稳定性热图。
- 特征值与目标的非线性关系图。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉非线性依赖。
- 不要求正态分布或线性关系。
- 可作为模型无关初筛方法。

### 局限

- 连续变量互信息估计可能不稳定。
- 不自动去除冗余特征。
- 得分没有像相关系数那样直观的方向。

### 常见坑

- 不区分连续和离散特征的处理方式。
- 在全数据上算互信息再做外部测试。
- 把互信息得分解释成风险比、回归系数或因果效应。

## 9. 与相近方法的区别

- 和 [[信息增益（Information Gain）]] 的区别：信息增益多用于分类树语境；互信息是更一般的变量依赖度量。
- 和 [[相关系数特征选择（Correlation-based Feature Selection）]] 的区别：相关系数主要捕捉线性或单调关系；互信息可捕捉更一般的依赖。
- 和 [[单变量特征选择（Univariate Feature Selection）]] 的区别：互信息是单变量特征选择的一种得分函数。

## 10. 医学研究中的典型应用

- 从非线性生物标志物候选集中筛选疾病相关变量。
- 影像组学特征初筛。
- 表格数据中寻找可能存在阈值效应的预测变量。

## 11. 相关方法

- [[信息增益（Information Gain）]]
- [[相关系数特征选择（Correlation-based Feature Selection）]]
- [[单变量特征选择（Univariate Feature Selection）]]
- [[K近邻算法（K-Nearest Neighbors, KNN）]]

## 12. 参考资料

- Cover TM, Thomas JA. *Elements of Information Theory*. 2nd ed. Wiley; 2006.
- Kraskov A, Stogbauer H, Grassberger P. Estimating mutual information. *Phys Rev E*. 2004;69(6):066138.
- scikit-learn Developers. `mutual_info_classif`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) （访问日期：2026-07-02）

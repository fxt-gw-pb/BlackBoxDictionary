---
title: 极端随机树
english_name: Extremely Randomized Trees
slug: extremely-randomized-trees
aliases: [extra trees, ExtraTrees, extremely randomized trees, "极端随机树（Extremely Randomized Trees）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 树模型, 随机森林]
status: 已建
difficulty: intermediate
question_type: 高随机性树集成预测
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ranger, extraTrees]
---

# 极端随机树（Extremely Randomized Trees）

## 1. 方法概览

### 1.1 定义

极端随机树，也称 Extra Trees，是一种树集成方法。它和随机森林相似，但在节点分裂时不仅随机抽取特征，还随机生成候选切分点，再从这些随机候选中选择较优切分。

### 1.2 它主要解决什么问题

- 研究问题：如何进一步降低树集成中单棵树之间的相关性和模型方差。
- 适用任务：二分类、多分类、连续结局预测。
- 常见医学场景：中高维临床特征预测、生物标志物筛选、表格数据稳健基线。

### 1.3 直觉理解

随机森林是在“随机看一部分特征后找最佳切分”，Extra Trees 则进一步把切分点也随机化。单棵树会更粗糙，但树与树之间更不相似，集成后可能更稳定、更快。

## 2. 数学形式

### 2.1 核心公式

若训练 $B$ 棵极端随机树 $T_b(x)$，分类预测为：

$$
\hat y=\arg\max_k\sum_{b=1}^{B}I(T_b(x)=k)
$$

回归预测为：

$$
\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

在某个节点上，对随机抽取的特征 $j$，Extra Trees 从该特征当前节点取值范围 $[a_j,b_j]$ 中随机抽取切分点：

$$
s_j\sim U(a_j,b_j)
$$

再按 Gini、熵或 MSE 减少量选择候选中最好的切分。

### 2.2 参数或统计量含义

- `n_estimators`：树的数量。
- `max_features`：每个节点参与随机候选的特征数量。
- `max_depth`：树深限制。
- `min_samples_leaf`：叶节点最小样本数。
- 随机切分点：Extra Trees 与随机森林的关键差异。

### 2.3 关键假设

- 数据中存在树模型可捕捉的非线性和交互。
- 通过更强随机性降低方差可以抵消单棵树偏差略升的问题。
- 目标以预测性能为主，而非单棵树解释。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：适合中高维表格数据。
- 是否适合缺失较多数据：通常需先处理缺失。
- 是否适合删失数据：原始方法不直接适合。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

以乳腺肿块良恶性分类为例：

| Radius | Texture | Perimeter | Area | Smoothness | Malignant |
| --- | --- | --- | --- | --- | --- |
| 17.9 | 10.4 | 122.8 | 1001 | 0.118 | 1 |
| 12.3 | 15.7 | 82.6 | 477 | 0.089 | 0 |
| 19.7 | 21.3 | 130.0 | 1203 | 0.110 | 1 |
| 11.4 | 14.9 | 73.5 | 402 | 0.082 | 0 |
| 15.1 | 18.2 | 99.1 | 712 | 0.101 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：树数、随机特征数、树深、叶节点最小样本数。
- 需要预处理的内容：缺失处理、类别编码、训练测试划分。

#### 产出

- 模型对象/统计结果：极端随机树集成、特征重要性。
- 参数估计：不提供传统回归系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：交叉验证性能、测试集 AUC / MSE、校准指标。

## 4. 适用场景

- 适合：需要快速训练、多特征、非线性关系明显且希望降低方差的任务。
- 不适合：强解释导向、样本极小或变量效应需要精确估计的任务。
- 使用前需要特别检查的点：是否过度随机导致欠拟合、特征重要性是否稳定、概率是否校准。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast_mass.csv")
X = df[["Radius", "Texture", "Perimeter", "Area", "Smoothness"]]
y = df["Malignant"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fit = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
fit.fit(X_train, y_train)

pred_prob = fit.predict_proba(X_test)[:, 1]
print(fit.feature_importances_)
```

### 5.2 R

常用包：

- `ranger`

```r
library(ranger)

fit <- ranger(
  Malignant ~ Radius + Texture + Perimeter + Area + Smoothness,
  data = df_train,
  probability = TRUE,
  num.trees = 500,
  splitrule = "extratrees",
  importance = "impurity"
)

pred <- predict(fit, data = df_test)$predictions
```

## 6. 结果如何解释

- 核心结果看什么：测试集性能、变量重要性、校准情况。
- 每个主要参数如何解释：`max_features` 控制节点随机候选特征数；叶节点越大模型越平滑。
- 临床或医学意义如何表达：可作为稳健预测模型或变量筛选初探，不应把重要性直接解释为因果效应。
- 常见误读：Extra Trees 训练快不代表一定比随机森林更准确。

## 7. 推荐可视化

- 特征重要性条形图。
- ROC / PR 曲线。
- 不同树数下交叉验证性能曲线。

## 8. 优势、局限与常见坑

### 优势

- 训练速度通常快于随机森林。
- 更强随机性可进一步降低方差。
- 对非线性和交互建模能力强。

### 局限

- 偏差可能高于随机森林。
- 可解释性仍然有限。
- 概率输出常需校准。

### 常见坑

- 不调 `min_samples_leaf` 导致概率不稳定。
- 只看 impurity importance，忽视其对连续变量或高基数变量的偏倚。
- 在小样本中使用过多复杂树后只看内部验证结果。

## 9. 与相近方法的区别

- 和 [[随机森林（Random Forest）]] 的区别：随机森林寻找候选特征中的最优切分点；Extra Trees 随机生成切分点后再选择。
- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：Extra Trees 是树模型上的强随机化集成。
- 和 [[决策树（Decision Tree）]] 的区别：Extra Trees 通过多树平均或投票降低单树不稳定性。

## 10. 医学研究中的典型应用

- 乳腺癌、肺结节等表格化影像特征分类。
- 多指标临床风险预测。
- 基因或蛋白标志物初筛中的非线性模型基线。

## 11. 相关方法

- [[随机森林（Random Forest）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[决策树（Decision Tree）]]
- [[随机森林回归（Random Forest Regression）]]

## 12. 参考资料

- Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. *Mach Learn*. 2006;63:3-42.
- scikit-learn Developers. `sklearn.ensemble.ExtraTreesClassifier`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) （访问日期：2026-07-02）
- Wright MN, Ziegler A. ranger: A fast implementation of random forests for high dimensional data in C++ and R. *J Stat Softw*. 2017;77(1):1-17.

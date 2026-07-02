---
title: 信息增益
english_name: Information Gain
slug: information-gain
aliases: [information gain, IG, "信息增益（Information Gain）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 信息论, 决策树]
status: 已建
difficulty: intermediate
question_type: 分类特征的信息量评估
data_type: [表格数据, 离散特征]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn, pandas]
r_packages: [FSelectorRcpp]
---

# 信息增益（Information Gain）

## 1. 方法概览

### 1.1 定义

信息增益是基于信息熵的特征评价指标，用来衡量使用某个特征划分数据后，类别不确定性减少了多少。它是 ID3 决策树和许多过滤式分类特征选择方法的核心思想。

### 1.2 它主要解决什么问题

- 研究问题：哪个特征最能减少分类结局的不确定性。
- 适用任务：离散特征筛选、分类树分裂特征选择、分类任务初筛。
- 常见医学场景：症状、分型、分组变量对疾病分类的信息量评估。

### 1.3 直觉理解

一个好特征能把混杂的样本分成更纯的组。信息增益衡量的就是“用了这个特征之后，我们对类别更确定了多少”。

## 2. 数学形式

### 2.1 核心公式

数据集 $D$ 的类别熵为：

$$
H(D)=-\sum_{k=1}^{K}p_k\log_2 p_k
$$

若特征 $A$ 有 $V$ 个取值，并将 $D$ 划分为 $D_1,\dots,D_V$，条件熵为：

$$
H(D\mid A)=\sum_{v=1}^{V}\frac{|D_v|}{|D|}H(D_v)
$$

信息增益为：

$$
IG(D,A)=H(D)-H(D\mid A)
$$

### 2.2 参数或统计量含义

- $H(D)$：划分前类别不确定性。
- $H(D\mid A)$：按特征 $A$ 划分后的剩余不确定性。
- $IG(D,A)$：不确定性减少量。
- 多值偏倚：取值很多的特征容易得到较高信息增益。

### 2.3 关键假设

- 目标变量为分类变量。
- 特征通常需要是离散变量；连续变量需先离散化或寻找切分点。
- 每个特征单独评价，不考虑多特征组合。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：离散特征、分箱后的连续特征。
- 因变量类型：二分类或多分类。
- 数据结构：分类监督学习数据。
- 是否适合高维数据：可用于高维离散特征筛选。
- 是否适合缺失较多数据：需设定缺失类别或先插补。
- 是否适合删失数据：不直接适合。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

以症状组合预测感染类型为例：

| Fever | Cough | CRP_high | WBC_high | Bacterial |
| --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 |
| 1 | 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 | 1 |
| 0 | 1 | 1 | 0 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：离散特征矩阵和分类标签。
- 关键变量：特征取值、类别分布、离散化策略。
- 需要预处理的内容：连续变量分箱、缺失处理、低频类别合并。

#### 产出

- 模型对象/统计结果：每个特征的信息增益得分。
- 参数估计：无。
- 预测结果：无，除非用于决策树模型。
- 不确定性指标：通常不提供置信区间，可用重采样评估稳定性。

## 4. 适用场景

- 适合：离散特征、分类目标、需要快速排序特征信息量的场景。
- 不适合：连续变量未合理分箱、多值特征很多、强交互主导的场景。
- 使用前需要特别检查的点：多值特征偏倚、分箱方式、类别不平衡。

## 5. 实现

### 5.1 Python

常用包：

- `pandas`
- `scikit-learn`

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv("infection_features.csv")
X = df[["Fever", "Cough", "CRP_high", "WBC_high"]]
y = df["Bacterial"].astype(int)

# 对离散变量，互信息得分可作为信息论特征排序的常用实现。
scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
ig_like = pd.Series(scores, index=X.columns).sort_values(ascending=False)

print(ig_like)
```

### 5.2 R

常用包：

- `FSelectorRcpp`

```r
library(FSelectorRcpp)

scores <- information_gain(
  Bacterial ~ Fever + Cough + CRP_high + WBC_high,
  data = df_train
)

scores[order(scores$importance, decreasing = TRUE), ]
```

## 6. 结果如何解释

- 核心结果看什么：哪些特征能最大幅度减少类别熵。
- 每个主要参数如何解释：信息增益越大，单独按该特征划分后类别越纯。
- 临床或医学意义如何表达：该特征对分类标签的信息量较高，但不代表因果作用。
- 常见误读：信息增益高可能只是因为取值很多或分箱过细。

## 7. 推荐可视化

- 信息增益得分条形图。
- 特征取值与类别分布堆叠条形图。
- 不同分箱方案下得分稳定性图。

## 8. 优势、局限与常见坑

### 优势

- 信息论基础清楚。
- 对离散分类任务直观。
- 可解释为不确定性减少。

### 局限

- 偏向取值多的特征。
- 连续变量需离散化。
- 忽略特征间冗余和交互。

### 常见坑

- 对连续变量随意分箱导致结果不稳定。
- 用信息增益筛出的特征直接当作独立危险因素。
- 不处理低频类别，导致熵估计不稳。

## 9. 与相近方法的区别

- 和 [[互信息特征选择（Mutual Information Feature Selection）]] 的区别：信息增益可视为目标熵减少量；互信息是更一般的两个变量共享信息量度。
- 和 [[ID3决策树（ID3 Decision Tree）]] 的区别：ID3 用信息增益选择树节点，本卡强调特征选择指标本身。
- 和 [[单变量特征选择（Univariate Feature Selection）]] 的区别：信息增益是单变量特征选择的一种信息论得分。

## 10. 医学研究中的典型应用

- 离散症状、体征、检查阳性指标的分类信息量排序。
- 临床规则树构建前的候选变量评估。
- 高维离散编码特征的初筛。

## 11. 相关方法

- [[互信息特征选择（Mutual Information Feature Selection）]]
- [[单变量特征选择（Univariate Feature Selection）]]
- [[ID3决策树（ID3 Decision Tree）]]
- [[C4.5决策树（C4.5 Decision Tree）]]

## 12. 参考资料

- Quinlan JR. Induction of decision trees. *Mach Learn*. 1986;1:81-106.
- Cover TM, Thomas JA. *Elements of Information Theory*. 2nd ed. Wiley; 2006.
- FSelectorRcpp Contributors. FSelectorRcpp package documentation. [https://cran.r-project.org/package=FSelectorRcpp](https://cran.r-project.org/package=FSelectorRcpp) （访问日期：2026-07-02）

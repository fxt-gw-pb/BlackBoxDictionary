---
title: 单变量特征选择
english_name: Univariate Feature Selection
slug: univariate-feature-selection
aliases: [univariate feature selection, SelectKBest, "单变量特征选择（Univariate Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 单变量分析]
status: 已建
difficulty: basic
question_type: 按特征与结局的单独关联筛选变量
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [FSelectorRcpp, caret]
---

# 单变量特征选择（Univariate Feature Selection）

## 1. 方法概览

### 1.1 定义

单变量特征选择是一类过滤式方法。它逐个评估每个特征与目标变量之间的关系，根据统计量、P 值或信息量得分选择排名靠前或达到阈值的特征。

### 1.2 它主要解决什么问题

- 研究问题：如何在大量候选特征中快速筛出与结局有单独关联的变量。
- 适用任务：分类、回归、高维数据初筛。
- 常见医学场景：影像组学特征初筛、组学标志物候选筛选、临床预测模型建模前变量缩减。

### 1.3 直觉理解

单变量特征选择像给每个候选变量单独面试：每次只看一个变量和结局的关系强不强。它快、直观，但看不到变量之间的组合效应和冗余关系。

## 2. 数学形式

### 2.1 核心公式

分类任务中，连续特征和分类结局常用 ANOVA F 统计量：

$$
F=\frac{\text{组间方差}}{\text{组内方差}}
$$

分类特征和分类结局常用卡方统计量：

$$
\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}
$$

连续结局回归中，可使用单变量线性回归的 F 统计量或相关系数。也可使用互信息：

$$
I(X;Y)=\sum_x\sum_y p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

### 2.2 参数或统计量含义

- 得分函数：如 `f_classif`、`chi2`、`f_regression`、`mutual_info_classif`。
- $k$：选择得分最高的前 $k$ 个特征。
- P 值阈值：按显著性筛选特征时使用。
- 多重检验：当特征很多时需要控制 FDR 或 FWER。

### 2.3 关键假设

- 每个特征可单独与结局评估。
- 统计检验的适用前提基本满足，例如卡方检验要求频数合理。
- 特征选择应在训练数据内部完成，避免信息泄露。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、分类变量、编码后的高维特征。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：适合初筛，但需多重检验控制。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：普通实现不直接适合删失结局。
- 是否适合重复测量数据：不直接适合，需避免同一患者信息泄露。

### 3.2 示例表格

以心血管事件预测的候选特征初筛为例：

| Age | SBP | LDL | CRP | Smoking | Event |
| --- | --- | --- | --- | --- | --- |
| 71 | 158 | 3.8 | 4.1 | 1 | 1 |
| 52 | 126 | 2.4 | 1.2 | 0 | 0 |
| 63 | 144 | 3.1 | 2.8 | 1 | 1 |
| 44 | 118 | 2.0 | 0.9 | 0 | 0 |
| 68 | 152 | 3.5 | 3.6 | 0 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：得分函数、选择数量 $k$ 或阈值。
- 需要预处理的内容：缺失处理、编码、必要时标准化、多重检验校正。

#### 产出

- 模型对象/统计结果：每个特征的分数、P 值或信息量。
- 参数估计：无模型参数，只有特征得分。
- 预测结果：无，输出筛选后的特征矩阵。
- 不确定性指标：P 值、校正后 P 值或交叉验证性能。

## 4. 适用场景

- 适合：特征数量很多、需要快速初筛、希望保留可解释的单变量关联线索。
- 不适合：变量间交互强、特征高度相关、需要寻找组合特征的场景。
- 使用前需要特别检查的点：选择过程是否嵌入交叉验证；是否控制多重比较；是否把单变量显著性误解为因果证据。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

df = pd.read_csv("cardio_event.csv")
X = df[["Age", "SBP", "LDL", "CRP", "Smoking"]]
y = df["Event"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

selector = SelectKBest(score_func=f_classif, k=3)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
print(selected_features.tolist())
print(scores)
```

### 5.2 R

常用包：

- `FSelectorRcpp`

```r
library(FSelectorRcpp)

scores <- information_gain(Event ~ Age + SBP + LDL + CRP + Smoking, data = df_train)
scores <- scores[order(scores$importance, decreasing = TRUE), ]

selected <- rownames(scores)[1:3]
x_train_sel <- df_train[, selected]
```

## 6. 结果如何解释

- 核心结果看什么：特征得分排序、P 值或校正后 P 值、选择后的模型性能。
- 每个主要参数如何解释：$k$ 越大保留越宽松；P 值阈值越小筛选越严格。
- 临床或医学意义如何表达：可以说某特征与结局有单独关联，但不能说它是独立预测因子或因果因素。
- 常见误读：单变量显著不代表多变量模型中仍重要；单变量不显著也不代表在组合模型中无用。

## 7. 推荐可视化

- 特征得分条形图。
- P 值火山图或排序图。
- 选择特征数量与交叉验证性能曲线。

## 8. 优势、局限与常见坑

### 优势

- 简单、快速、易解释。
- 适合高维数据第一轮筛选。
- 可按特征类型选择不同统计检验。

### 局限

- 忽略特征之间的相关性和交互。
- 容易受多重检验影响。
- 可能筛掉弱单变量但强组合效应的变量。

### 常见坑

- 在全数据上做单变量筛选，再划分训练测试集。
- 用未经校正的 P 值筛选大量特征。
- 把单变量筛选结果写成“独立危险因素”。

## 9. 与相近方法的区别

- 和 [[方差阈值法（Variance Threshold）]] 的区别：单变量方法使用结局变量；方差阈值不使用结局。
- 和 [[互信息特征选择（Mutual Information Feature Selection）]] 的区别：互信息是单变量特征选择的一种非线性得分函数。
- 和 [[嵌入式特征选择（Embedded Feature Selection）]] 的区别：嵌入式方法在模型训练中选择特征。

## 10. 医学研究中的典型应用

- 影像组学中从数百个纹理特征中初筛候选特征。
- 组学数据中筛选与疾病状态相关的基因或蛋白。
- 临床预测模型建模前的候选变量缩减。

## 11. 相关方法

- [[互信息特征选择（Mutual Information Feature Selection）]]
- [[信息增益（Information Gain）]]
- [[多重检验与错误率控制（Multiple Testing and Error Rate Control）]]
- [[Benjamini-Hochberg程序（Benjamini-Hochberg Procedure）]]

## 12. 参考资料

- Guyon I, Elisseeff A. An introduction to variable and feature selection. *J Mach Learn Res*. 2003;3:1157-1182.
- scikit-learn Developers. Univariate feature selection. scikit-learn User Guide. [https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) （访问日期：2026-07-02）
- Kuhn M, Johnson K. *Feature Engineering and Selection: A Practical Approach for Predictive Models*. Chapman and Hall/CRC; 2019.

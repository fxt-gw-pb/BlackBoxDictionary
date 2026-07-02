---
title: 方差阈值法
english_name: Variance Threshold
slug: variance-threshold
aliases: [variance threshold, low variance filter, "方差阈值法（Variance Threshold）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 数据预处理]
status: 已建
difficulty: basic
question_type: 低信息量特征过滤
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督预处理, 多种]
python_packages: [scikit-learn]
r_packages: [caret, recipes]
---

# 方差阈值法（Variance Threshold）

## 1. 方法概览

### 1.1 定义

方差阈值法是一种过滤式特征选择方法。它计算每个特征在样本中的方差，并删除方差低于预设阈值的特征，用来去掉几乎不变化、信息量很低的变量。

### 1.2 它主要解决什么问题

- 研究问题：如何在建模前快速删除近似常量特征。
- 适用任务：无监督特征过滤、建模前降噪、高维矩阵预处理。
- 常见医学场景：组学特征初筛、影像组学稳定特征过滤、电子病历中几乎恒定字段清理。

### 1.3 直觉理解

如果一个变量在所有患者中几乎一样，它很难帮助模型区分不同个体。方差阈值法就是先把这类“没有变化”的列清掉，让后续模型少背一些没有信息的包袱。

## 2. 数学形式

### 2.1 核心公式

对第 $j$ 个特征 $X_j$，其样本方差为：

$$
s_j^2=\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\bar x_j)^2
$$

设阈值为 $\theta$，选择规则为：

$$
\text{keep}(X_j)=I(s_j^2\geq \theta)
$$

对于二分类哑变量，若取值为 1 的比例为 $p$，方差为：

$$
\mathrm{Var}(X_j)=p(1-p)
$$

### 2.2 参数或统计量含义

- $s_j^2$：第 $j$ 个特征的样本方差。
- $\theta$：方差阈值。
- 近零方差：特征几乎恒定，或某个取值占绝大多数。
- `threshold`：实现中的阈值参数。

### 2.3 关键假设

- 特征变化越小，通常提供的信息越少。
- 这是无监督过滤方法，不使用结局变量。
- 只有在特征尺度可比时，方差大小才有直接比较意义。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、二值变量、编码后的分类变量。
- 因变量类型：不需要结局变量。
- 数据结构：样本乘以特征的矩阵。
- 是否适合高维数据：适合，尤其适合作为第一步粗筛。
- 是否适合缺失较多数据：需先处理缺失或按规则计算方差。
- 是否适合删失数据：作为特征预处理可用，但不处理删失结局。
- 是否适合重复测量数据：需先明确分析单位，避免把同一患者重复记录当作独立样本。

### 3.2 示例表格

以 ICU 表格特征预处理为例：

| Age | SexMale | HospitalID_01 | RareCode_X | Lactate | Death30d |
| --- | --- | --- | --- | --- | --- |
| 73 | 1 | 1 | 0 | 3.1 | 1 |
| 46 | 0 | 1 | 0 | 1.1 | 0 |
| 65 | 1 | 1 | 0 | 2.4 | 1 |
| 39 | 0 | 1 | 0 | 0.8 | 0 |
| 58 | 1 | 1 | 0 | 1.6 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵。
- 关键变量：每个特征的方差、阈值。
- 需要预处理的内容：缺失处理、类别编码、是否标准化的决策。

#### 产出

- 模型对象/统计结果：保留特征掩码、各特征方差。
- 参数估计：无。
- 预测结果：无，输出降维后的特征矩阵。
- 不确定性指标：通常不提供。

## 4. 适用场景

- 适合：删除常量列、近似常量列、极低频哑变量。
- 不适合：单独作为最终特征选择依据，尤其不适合判断特征与结局是否相关。
- 使用前需要特别检查的点：特征尺度是否可比；二值特征的低方差是否仍有医学意义，例如罕见但重要的突变。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

df = pd.read_csv("icu_features.csv")
X = df.drop(columns=["Death30d"])
y = df["Death30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

selector = VarianceThreshold(threshold=0.01)
X_train_sel = selector.fit_transform(X_train)
X_test_sel = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print(selected_features.tolist())
```

### 5.2 R

常用包：

- `caret`

```r
library(caret)

x_train <- df_train[, setdiff(names(df_train), "Death30d")]
nzv <- nearZeroVar(x_train)

x_train_sel <- x_train[, -nzv]
x_test_sel <- df_test[, names(x_train_sel)]
```

## 6. 结果如何解释

- 核心结果看什么：哪些特征被删除、删除原因是否只是低变化。
- 每个主要参数如何解释：阈值越高，删除越激进。
- 临床或医学意义如何表达：该方法只说明某特征在当前样本中变化很少，不说明其没有生物学或临床意义。
- 常见误读：低方差不等于无效，高方差也不等于有预测价值。

## 7. 推荐可视化

- 特征方差排序条形图。
- 低方差特征取值频数图。
- 阈值与保留特征数量关系图。

## 8. 优势、局限与常见坑

### 优势

- 简单、快速、无需结局变量。
- 适合高维数据的第一步粗筛。
- 能清理常量列和无效哑变量。

### 局限

- 不评估特征与结局关系。
- 对特征尺度敏感。
- 可能删除低频但临床重要的特征。

### 常见坑

- 标准化后所有连续特征方差都接近 1，再做方差阈值就失去意义。
- 在全数据上先筛选，再做训练测试划分，造成数据处理流程不规范。
- 对罕见事件变量机械删除。

## 9. 与相近方法的区别

- 和 [[单变量特征选择（Univariate Feature Selection）]] 的区别：方差阈值不看结局；单变量方法会评估每个特征与结局的关系。
- 和 [[相关系数特征选择（Correlation-based Feature Selection）]] 的区别：方差阈值删除低变化特征；相关系数法删除高冗余特征。
- 和 [[嵌入式特征选择（Embedded Feature Selection）]] 的区别：方差阈值不依赖模型训练。

## 10. 医学研究中的典型应用

- 删除所有患者取值完全相同的 EHR 字段。
- 影像组学中去除几乎不变化的纹理特征。
- 组学分析中先删除低变异探针或低表达特征。

## 11. 相关方法

- [[单变量特征选择（Univariate Feature Selection）]]
- [[相关系数特征选择（Correlation-based Feature Selection）]]
- [[嵌入式特征选择（Embedded Feature Selection）]]

## 12. 参考资料

- Kuhn M, Johnson K. *Feature Engineering and Selection: A Practical Approach for Predictive Models*. Chapman and Hall/CRC; 2019.
- scikit-learn Developers. `sklearn.feature_selection.VarianceThreshold`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html) （访问日期：2026-07-02）
- Kuhn M. Building predictive models in R using the caret package. *J Stat Softw*. 2008;28(5):1-26.

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

### 1.1 一句话本质

相关过滤把高度同步变化的特征视为重复测量，在每组冗余变量中保留代表，以降低共线性与重复信息。

### 1.2 定义

该方法计算特征—特征相关矩阵，按 $|r|$ 阈值删除高度相关列；有时也按特征—结局相关排序。两种用途必须区分：前者是无监督去冗余，后者是单变量监督筛选。

### 1.3 它主要解决什么问题

- 多个仪器读数、派生指标或组学特征表达几乎同一信号。
- 减少线性模型系数不稳定和距离模型的重复计权。
- 医学场景：影像组学、实验室同义指标、连续波形派生特征。

### 1.4 直觉与类比

摄氏温度与华氏温度携带同一信息。把两者都送入模型不会增加知识，却可能让算法把“温度”重复计算两次；相关过滤保留其中一个代表。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

高维特征工程会产生大量线性重复。回归系数可能在相关变量间任意分摊，重要性排名不稳；KNN/PCA 等也会让某个潜在概念因多列复制而权重过大。

### 2.2 关键洞察

若两个特征几乎可由线性变换互相预测，保留二者的增量信息很少。删除哪一个不应只看相关系数，还应看缺失率、测量成本、可解释性、与结局的关系及未来可获得性。

### 2.3 与朴素/相邻做法的对比

- VIF 考察一列被其余所有列联合解释的程度；成对相关只看两两关系。
- Lasso/Elastic Net 在监督模型中处理共线性，但选择可能随样本波动。
- Spearman 可发现单调非线性冗余；Pearson 主要度量线性关系。

## 3. 数学形式

### 3.1 核心公式

Pearson 相关：

$$
r_{jk}=
\frac{\sum_i(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)}
{\sqrt{\sum_i(x_{ij}-\bar x_j)^2
\sum_i(x_{ik}-\bar x_k)^2}}
$$

简单删除规则：

$$
|r_{jk}|>\tau\quad\Rightarrow\quad
\text{从 }j,k\text{ 中删除一个}
$$

这个式子在说：两列标准化后的共同变化越接近完全同步或反向同步，冗余越高。

### 3.2 推导脉络

相关是标准化协方差，因此不受线性单位换算影响。绝对值处理正、负重复；但 $r\approx0$ 只表示无线性关系，不代表独立。

### 3.3 参数与统计量含义

- $r_{jk}$：两特征线性冗余强度，范围 $[-1,1]$。
- $\tau$：常取 0.8–0.95，但应做敏感性分析。
- clustering/linkage：将相关矩阵转成距离后分组的规则。
- representative：每组最终保留的代表特征。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后果 |
| --- | --- | --- |
| 相关估计稳定 | 样本量足够、异常值可控 | 错分冗余组 |
| 线性/单调冗余是重点 | 非线性重复不重要 | 漏掉复杂冗余 |
| 训练集拟合筛选 | 测试不参与矩阵估计 | 泄漏 |
| 删除规则符合部署 | 保留列未来可获得 | 上线缺列或成本过高 |

## 4. 手把手算例

6 人特征 A 为 $(1,2,3,4,5,6)$，B 为 $(2,4,6,8,10,12)=2A$，结局 $Y=(1,2,2,4,5,6)$。

因为 B 是 A 的精确线性变换：

$$
r_{AB}=1
$$

A 的离均差平方和为 17.5；Y 的为 $19.333$，交叉乘积和为 18：

$$
r_{AY}=\frac{18}{\sqrt{17.5\times19.333}}=0.979
$$

线性缩放不改变相关，所以 $r_{BY}=0.979$。阈值 $\tau=0.9$ 时 A、B 属于同一冗余组；二者对结局排名完全相同。若 A 缺失率 2%、B 缺失率 20%，应保留 A，而不是随意删除第一列或最后一列。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 连续或有序表格特征；类别变量需专门关联度量。
- 对缺失、批次、中心差异和重复测量需额外处理。
- 不处理删失、混杂或因果关系。

### 5.2 示例表格

| patient_id | marker_A | marker_B | outcome |
| --- | ---: | ---: | ---: |
| P01 | 1 | 2 | 1 |
| P02 | 2 | 4 | 2 |
| P03 | 3 | 6 | 2 |

### 5.3 输入与产出

输入训练特征矩阵、相关类型、阈值与代表选择规则；产出相关矩阵、冗余组和保留列。

## 6. 适用场景

- 适合：影像组学、派生特征多、共线性明显。
- 不适合：类别变量为主、非线性冗余核心、删除任一特征都会破坏解释组合。
- 不应只按与结局相关大小决定一切，以免泄漏和过拟合。

## 7. 实现

### 7.1 Python

```python
import numpy as np
import pandas as pd

X = pd.DataFrame({
    "A": [1, 2, 3, 4, 5, 6],
    "B": [2, 4, 6, 8, 10, 12],
    "C": [0, 1, 0, 1, 0, 1],
})
corr = X.corr(method="pearson").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop = [col for col in upper.columns if (upper[col] > 0.9).any()]
print(corr, drop)
```

### 7.2 R

```r
library(caret)

X <- data.frame(
  A = 1:6,
  B = 2 * (1:6),
  C = c(0, 1, 0, 1, 0, 1)
)
corr <- cor(X, use = "pairwise.complete.obs")
drop_index <- findCorrelation(corr, cutoff = 0.9)
names(X)[drop_index]
```

## 8. 结果如何解读

被删除表示可由保留列近似线性替代，不表示生物学无意义。应记录每组相关结构、删除规则和代表变量，保证训练与部署使用同一列集合。

## 9. 假设诊断与稳健性

- 用 bootstrap/交叉验证检查相关和保留列稳定性。
- 比较 Pearson、Spearman 与 biweight midcorrelation。
- 对异常值、批次和亚组分别画相关矩阵。
- 扫描多个阈值并评估下游性能。
- 相关过滤后仍可检查 VIF 或条件数。

## 10. 推荐可视化

- 聚类相关热图。
- 特征相关网络图。
- 冗余组散点图与异常值标注。
- 阈值—保留数量—验证性能曲线。

## 11. 优势、局限与常见坑

### 优势

- 简单快速，结果直观。
- 与结局无关的去冗余可降低直接监督泄漏。

### 局限

- 只捕捉选定相关类型的关系。
- 删除代表规则可能任意且不稳定。

### 常见坑

- 在全数据上计算相关矩阵。
- 忽略负相关而不用绝对值。
- 用 pairwise complete 造成每对样本基础不同。
- 把低相关当独立或因果无关。

## 12. 与相近方法的区别

- 方差阈值删除单列无变化；相关过滤删除列间重复。
- 单变量选择按结局排序，可能同时保留多个冗余列。
- PCA 用线性组合压缩，相关过滤保留原变量语义。
- 如何选择：重视解释时优先相关分组留代表；重视压缩可比较 PCA。

## 13. 医学研究中的典型应用

- 影像组学纹理特征去冗余。
- 高相关实验室指标和生命体征摘要。
- 组学共表达特征的代表选择。

应说明缺失处理、批次/中心、阈值、代表规则和患者级切分。

## 14. 关键术语

- **冗余（Redundancy）**：一个特征可被另一个近似替代。
- **共线性（Collinearity）**：预测变量间存在强线性关系。
- **Pearson 相关**：线性共同变化的标准化度量。
- **Spearman 相关**：基于秩的单调关系度量。
- **VIF**：一列被其余列联合解释的膨胀程度。

## 15. 相关方法

- [[Pearson相关（Pearson Correlation）]]
- [[Spearman秩相关（Spearman Rank Correlation）]]
- [[方差阈值法（Variance Threshold）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Lasso回归（Lasso Regression）]]

## 16. 参考资料

- Kuhn M, Johnson K. *Applied Predictive Modeling*. Springer; 2013.
- Guyon I, Elisseeff A. An introduction to variable and feature selection. *JMLR*. 2003;3:1157-1182.

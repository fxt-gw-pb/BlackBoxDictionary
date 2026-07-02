---
title: 层次聚类
english_name: Hierarchical Clustering
slug: hierarchical-clustering
aliases: [hierarchical clustering, agglomerative clustering, 凝聚层次聚类, "层次聚类（Hierarchical Clustering）"]
category: 聚类与无监督学习
subcategory: 层次聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 树状图]
status: 已建
difficulty: basic
question_type: 层次结构分群与样本相似性探索
data_type: [表格数据, 距离矩阵, 高维特征矩阵]
outcome_type: [无监督分群]
python_packages: [scipy, scikit-learn]
r_packages: [stats, cluster]
---

# 层次聚类（Hierarchical Clustering）

## 1. 方法概览

### 1.1 定义

层次聚类是一类把样本组织成树状层次结构的无监督方法。最常见的是凝聚式层次聚类：先把每个样本看作一个簇，再按相似性逐步合并，最终形成一棵树状图。

### 1.2 它主要解决什么问题

- 研究问题：样本之间是否存在从细到粗的层级分群结构。
- 适用任务：探索性分群、样本相似性可视化、基因或患者层次结构展示。
- 常见医学场景：热图行列聚类，组学样本分层，症状或表型相似性结构分析。

### 1.3 直觉理解

层次聚类像是在不断把最相似的对象归并成更大的组。树状图的低处分叉表示样本很相似，高处分叉表示只有在较粗层次下才被合并。

## 2. 数学形式

### 2.1 核心公式

设两个簇为 $A$ 和 $B$，层次聚类需要定义簇间距离。常见 linkage 包括：

单链接：

$$
d(A,B)=\min_{x_i\in A,x_j\in B} d(x_i,x_j)
$$

完全链接：

$$
d(A,B)=\max_{x_i\in A,x_j\in B} d(x_i,x_j)
$$

平均链接：

$$
d(A,B)=\frac{1}{|A||B|}\sum_{x_i\in A}\sum_{x_j\in B}d(x_i,x_j)
$$

Ward 方法每次合并使簇内平方和增加最小：

$$
\Delta(A,B)=\sum_{x_i\in A\cup B}\|x_i-\mu_{A\cup B}\|^2
-\sum_{x_i\in A}\|x_i-\mu_A\|^2
-\sum_{x_i\in B}\|x_i-\mu_B\|^2
$$

### 2.2 参数或统计量含义

- 距离度量：样本之间的距离，如欧氏距离、曼哈顿距离、相关距离。
- linkage：簇间距离定义，如 single、complete、average、ward。
- cut height：树状图剪切高度，用于得到最终簇标签。
- cophenetic correlation：树状图距离与原始距离的一致性指标。

### 2.3 关键假设

- 所选距离能表达研究问题中的相似性。
- 层级结构对数据有意义。
- 合并策略与簇形状相匹配。
- 样本数不宜过大，否则树状图和距离矩阵都会变得沉重。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征、标准化特征或预先计算的距离矩阵。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：样本乘以特征矩阵，或样本乘以样本距离矩阵。
- 是否适合高维数据：可用，组学热图常见，但需注意距离稳定性。
- 是否适合缺失较多数据：需先处理缺失，或选择能处理缺失的距离。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先定义样本单位或构造个体级特征。

### 3.2 示例表格

以基因表达模块探索为例：

| GeneA | GeneB | GeneC | GeneD | Group |
| --- | --- | --- | --- | --- |
| 1.2 | 0.4 | -0.3 | 0.8 | Case |
| 1.0 | 0.5 | -0.2 | 0.7 | Case |
| -0.6 | 1.5 | 0.9 | -0.4 | Control |
| -0.4 | 1.2 | 0.8 | -0.5 | Control |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵或距离矩阵。
- 关键变量：距离度量、linkage、剪切高度或簇数。
- 需要预处理的内容：缺失处理、标准化、异常值检查、变量筛选。

#### 产出

- 模型对象/统计结果：树状图、合并矩阵、剪切后的簇标签。
- 参数估计：每一步合并的簇和距离。
- 预测结果：普通层次聚类不自然支持新样本快速预测。
- 不确定性指标：bootstrap 稳定性、cophenetic correlation、不同 linkage 对比。

## 4. 适用场景

- 适合：样本量中小、希望观察层级结构、需要树状图或热图展示的场景。
- 不适合：超大样本、需要在线更新、簇边界明显非层级或噪声点很多的场景。
- 使用前需要特别检查的点：距离度量、linkage 选择、标准化策略、树状图剪切是否稳定。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`
- `scikit-learn`

```python
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("omics_profiles.csv")
X = df[["GeneA", "GeneB", "GeneC", "GeneD"]]
X_scaled = StandardScaler().fit_transform(X)

Z = linkage(X_scaled, method="ward")
cluster = fcluster(Z, t=3, criterion="maxclust")

df["cluster"] = cluster
print(df["cluster"].value_counts().sort_index())
```

### 5.2 R

常用包：

- `stats`

```r
x <- scale(df[, c("GeneA", "GeneB", "GeneC", "GeneD")])
d <- dist(x, method = "euclidean")
fit <- hclust(d, method = "ward.D2")

cluster <- cutree(fit, k = 3)
table(cluster)
```

## 6. 结果如何解释

- 核心结果看什么：树状图分叉高度、主要分支、剪切后簇的特征模式。
- 每个主要参数如何解释：Ward 方法倾向得到相对紧凑、方差较小的簇。
- 临床或医学意义如何表达：可描述为“在所选特征和距离度量下，样本呈现若干层级相似性结构”。
- 常见误读：树状图分支不等同于系统发育关系或因果演化关系。

## 7. 推荐可视化

- 树状图。
- 带行列聚类的热图。
- 不同剪切高度下的簇数变化图。
- PCA/UMAP 上叠加层次聚类标签。

## 8. 优势、局限与常见坑

### 优势

- 不必先固定簇数，可以事后剪切树状图。
- 树状图直观展示层级关系。
- 适合与热图结合用于组学和临床特征展示。

### 局限

- 对距离和 linkage 选择敏感。
- 早期合并一旦发生通常不会被撤销。
- 大样本计算和可视化都较困难。

### 常见坑

- 混用不同尺度变量而不标准化。
- 只看树状图视觉效果，不评估稳定性。
- 随意剪切树状图并赋予过强医学意义。

## 9. 与相近方法的区别

- 和 [[K-means聚类（K-means Clustering）]] 的区别：K-means 直接给定 $K$ 个互斥簇，层次聚类提供从细到粗的树状结构。
- 和 [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]] 的区别：BIRCH 用 CF 树压缩数据，更适合大规模数据；传统层次聚类更适合中小样本解释。
- 和 [[谱聚类（Spectral Clustering）]] 的区别：谱聚类通过图拉普拉斯嵌入处理复杂形状簇，层次聚类主要依赖距离和合并策略。

## 10. 医学研究中的典型应用

- 基因表达热图中的样本和基因聚类。
- 患者临床表型层级相似性展示。
- 多组学特征模块或样本分支结构探索。

## 11. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]
- [[谱聚类（Spectral Clustering）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 12. 参考资料

- Murtagh F, Contreras P. Algorithms for hierarchical clustering: an overview. *WIREs Data Mining and Knowledge Discovery*. 2012;2(1):86-97.
- Ward JH Jr. Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*. 1963;58(301):236-244.
- SciPy Developers. Hierarchical clustering documentation. [https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) （访问日期：2026-07-02）

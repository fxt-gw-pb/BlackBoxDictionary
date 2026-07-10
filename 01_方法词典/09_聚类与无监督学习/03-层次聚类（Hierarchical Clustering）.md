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

### 1.1 一句话本质

层次聚类按距离逐步合并或拆分样本，把从细到粗的分群过程保存为一棵树，而不是只给单一分组。

### 1.2 定义

层次聚类构造嵌套簇结构。最常见的凝聚式方法从每个样本各自成簇开始，每轮合并距离最近的两簇，直到只剩一个簇；在树状图某一高度剪切即可得到平面标签。

### 1.3 它主要解决什么问题

- 样本是否存在多尺度、嵌套的相似性结构。
- 不想在建树前固定唯一簇数。
- 需要将聚类与热图结合展示患者或基因模式。

### 1.4 直觉与类比

像整理家谱：先合并最相似的个体，再把相似的小组并成更大的家族。分叉越低表示越相似，分叉越高表示只有在更粗尺度下才归在一起。

## 2. 核心思想与原理

### 2.1 两个选择决定结果

样本距离定义“谁像谁”，linkage 定义“两个簇如何比较”。相同数据改用相关距离、欧氏距离或不同 linkage，树可能明显变化。

### 2.2 合并不可撤销

凝聚算法一旦合并两簇，后续不会拆开。因此早期局部决定会影响整个树，这也是对噪声和距离选择敏感的原因。

### 2.3 树不是天然真相

任何距离矩阵都能生成一棵树。树状图存在不等于生物学上真有层级亚型，仍需稳定性和外部解释。

## 3. 数学形式

### 3.1 常见 linkage

单链接：

$$
d_{\mathrm{single}}(A,B)=
\min_{x\in A,z\in B}d(x,z)
$$

完全链接：

$$
d_{\mathrm{complete}}(A,B)=
\max_{x\in A,z\in B}d(x,z)
$$

平均链接：

$$
d_{\mathrm{average}}(A,B)=
\frac{1}{|A||B|}
\sum_{x\in A}\sum_{z\in B}d(x,z)
$$

### 3.2 Ward 合并代价

$$
\Delta(A,B)=
\operatorname{SSE}(A\cup B)
-\operatorname{SSE}(A)
-\operatorname{SSE}(B)
$$

Ward 每轮选择使簇内平方和增加最小的合并，通常与欧氏距离配合。

### 3.3 树的剪切

给定剪切高度 $h$，高度低于 $h$ 已连接的叶子归为同簇。也可直接指定最终簇数 $K$。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 距离符合研究含义 | 树反映错误相似性 | 比较距离定义 |
| 尺度处理合理 | 大量纲变量主导 | 标准化敏感性 |
| 层次结构相对稳定 | 分支随重采样改变 | bootstrap 共识 |
| 样本量可承受距离矩阵 | 内存和图形不可用 | 记录复杂度、先压缩 |

## 4. 手把手算例

一维有 4 个样本：

$$
A=1,\quad B=2,\quad C=8,\quad D=9
$$

使用欧氏距离。

**Step 1：计算最近样本。** $d(A,B)=1$，$d(C,D)=1$，其余距离至少为 6。先合并 $AB$，再合并 $CD$，两次高度均为 1。

**Step 2：比较两个大簇。**

单链接取最近点：

$$
d_{\mathrm{single}}(AB,CD)=d(B,C)=6
$$

完全链接取最远点：

$$
d_{\mathrm{complete}}(AB,CD)=d(A,D)=8
$$

平均链接取四个跨簇距离平均：

$$
d_{\mathrm{average}}
=\frac{7+8+6+7}{4}=7
$$

**Step 3：读树状图。** 三种 linkage 都先得到 $\{A,B\}$ 和 $\{C,D\}$，但最终合并高度分别为 6、8、7。

**结论：** linkage 不改变这组数据的两个明显小簇，却改变“两个簇相距多远”的定义；复杂数据中甚至可能改变整个合并顺序。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 可输入样本-特征矩阵或预计算距离矩阵。
- 连续特征常需标准化；组学中可按问题使用相关距离。
- 缺失需在距离计算前处理。

### 5.2 输入与产出

输入为距离、linkage 和剪切规则。输出为 linkage 矩阵、树状图、合并高度与剪切标签。传统方法不自然支持新样本在线分配。

## 6. 适用场景

- 中小样本，希望观察多层次结构或绘制聚类热图。
- 对样本或变量的相似关系比在线预测更重要。
- 不适合超大样本、噪声很多或必须快速给新样本标签的场景。

## 7. 实现

### 7.1 Python

```python
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X_s = StandardScaler().fit_transform(X)
Z = linkage(
    X_s,
    method="ward",
    metric="euclidean",
    optimal_ordering=True,
)
cluster = fcluster(Z, t=3, criterion="maxclust")

dendrogram(Z, labels=sample_ids)
plt.ylabel("merge distance")
plt.show()
```

### 7.2 R

```r
x <- scale(df[, c("GeneA", "GeneB", "GeneC", "GeneD")])
d <- dist(x, method = "euclidean")
fit <- hclust(d, method = "ward.D2")

cluster <- cutree(fit, k = 3)
plot(fit, labels = rownames(df), hang = -1)
rect.hclust(fit, k = 3, border = 2:4)
```

## 8. 结果如何解释

- 合并高度表示所选 linkage 下的差异尺度，不一定是原始两点距离。
- 叶子左右顺序并非唯一，邻近显示不一定意味着直接合并。
- 剪切高度或 $K$ 是分析选择，应结合稳定性与临床可解释性。
- 树状图不是演化树，也不代表时间或因果方向。

## 9. 诊断与稳健性

1. 比较 Euclidean、Manhattan 或相关距离。
2. 比较 single、complete、average 与 Ward。
3. 计算 cophenetic correlation 评估树对原距离的保真。
4. bootstrap 样本或特征，评估分支复现率。
5. 检查离群点是否在高处单独合并并扭曲树。

## 10. 推荐可视化

- 标注合并高度的树状图。
- 行列双向聚类热图。
- 不同 linkage 的树状图并排对照。
- bootstrap 共识矩阵或分支稳定性图。

## 11. 优势、局限与常见坑

**优势：** 不必建树前固定簇数，完整展示多尺度层次，适合热图。

**局限：** 距离与 linkage 敏感，合并不可撤销，计算和存储通常近二次增长。

**常见坑：** 混用不同尺度；把叶序当距离；随意剪树；只展示漂亮热图不做稳定性；将树解释为生物演化。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：直接优化固定 $K$ 的平面分组。
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]：用 CF 树压缩大数据，更适合扩展。
- [[谱聚类（Spectral Clustering）]]：基于图拉普拉斯嵌入处理非凸结构。
- 选择经验：需要树状层级且样本量中小时使用传统层次聚类。

## 13. 医学研究中的典型应用

- 基因表达、蛋白组和代谢组热图。
- 患者表型或症状模式的层次探索。
- 基因、细胞或特征模块的相似性组织。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| dendrogram | 记录逐步合并关系和高度的树状图 |
| linkage | 定义两个簇之间距离的规则 |
| cut height | 将树转换为平面簇的剪切高度 |
| Ward method | 最小化簇内平方和增量的合并法 |
| cophenetic distance | 两个叶子首次合并时的树高度 |

## 15. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]
- [[谱聚类（Spectral Clustering）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- Ward JH Jr. Hierarchical grouping to optimize an objective function. *J Am Stat Assoc*. 1963;58(301):236-244.
- Murtagh F, Contreras P. Algorithms for hierarchical clustering: an overview. *WIREs Data Min Knowl Discov*. 2012;2(1):86-97.
- SciPy Developers. Hierarchical clustering documentation. [https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html) （访问日期：2026-07-09）

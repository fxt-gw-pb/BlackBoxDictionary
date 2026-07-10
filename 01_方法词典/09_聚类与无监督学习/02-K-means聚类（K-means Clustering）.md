---
title: K-means聚类
english_name: K-means Clustering
slug: k-means-clustering
aliases: [K-means, k-means clustering, K均值聚类, "K-means聚类（K-means Clustering）"]
category: 聚类与无监督学习
subcategory: 原型聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 原型方法]
status: 已建
difficulty: basic
question_type: 硬聚类与患者亚群探索
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [stats, factoextra]
---

# K-means聚类（K-means Clustering）

## 1. 方法概览

### 1.1 一句话本质

K-means 把每个样本分给最近的质心，并反复更新质心，使总簇内平方距离尽可能小。

### 1.2 定义

K-means 是基于质心的硬聚类方法。用户预先指定 $K$，算法交替进行最近质心分配与簇内均值更新，直到标签或目标函数基本不再变化。

### 1.3 它主要解决什么问题

- 无标签数值数据能否分为若干紧凑、互斥的亚群。
- 用少量原型概括大量患者。
- 为后续描述、采样或可视化生成探索性分组。

### 1.4 直觉与类比

在患者特征空间放置 $K$ 个“典型患者”。每个人归到最像的典型患者名下，再用组内平均重新定义典型患者，反复直到稳定。

## 2. 核心思想与原理

### 2.1 交替最小化

固定质心时，最近质心分配最优；固定标签时，算术均值使平方距离和最小。两个步骤都不增加目标函数，但只保证收敛到局部最优。

### 2.2 隐含的簇形状

欧氏距离与均值原型使 K-means 偏好近似球形、尺度相近、方差相近的簇。弯月形、长条形或密度差异大的簇可能被错误切开。

### 2.3 为什么标准化重要

平方距离会放大量纲。SBP 的数值范围若远大于 HDL，未标准化时聚类可能几乎只由 SBP 决定。

## 3. 数学形式

### 3.1 目标函数

$$
\operatorname*{minimize}_{C_1,\ldots,C_K}
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|_2^2
$$

### 3.2 分配与更新

$$
c_i=
\operatorname*{arg\,min}_{k}
\|x_i-\mu_k\|_2^2
$$

$$
\mu_k=
\frac{1}{|C_k|}
\sum_{x_i\in C_k}x_i
$$

### 3.3 评价量

- inertia/WCSS：目标函数值，随 $K$ 增大必然不升。
- silhouette：比较样本簇内紧密度与最近其他簇距离。
- `n_init`：不同初始质心重复次数，取目标最小解。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 欧氏距离有意义 | 分组不反映临床相似性 | 变量与距离审查 |
| 特征尺度可比 | 大量纲变量支配 | 标准化敏感性分析 |
| 簇近似紧凑凸形 | 非球形结构被切错 | 二维投影与替代算法 |
| 离群点有限 | 质心被拉偏 | 稳健预处理与病例审查 |

## 4. 手把手算例

一维数据为：

$$
x=(1,2,8,9)
$$

设 $K=2$，初始质心 $\mu_1=1,\mu_2=8$。

**Step 1：分配。**

- 1 和 2 更接近 $\mu_1$。
- 8 和 9 更接近 $\mu_2$。

得到 $C_1=\{1,2\}$、$C_2=\{8,9\}$。

**Step 2：更新质心。**

$$
\mu_1=\frac{1+2}{2}=1.5,\qquad
\mu_2=\frac{8+9}{2}=8.5
$$

**Step 3：重新分配。** 四个点所属簇不变，因此算法收敛。

**Step 4：计算 WCSS。**

$$
\begin{aligned}
\operatorname{WCSS}
&=(1-1.5)^2+(2-1.5)^2\\
&\quad +(8-8.5)^2+(9-8.5)^2\\
&=1
\end{aligned}
$$

**结论：** K-means 给出互斥标签和两个原型 1.5、8.5，不表达边界病例属于两簇的概率。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 主要用于连续数值特征。
- 缺失需预先处理，类别变量不能简单当连续数值编码。
- 重复测量应先定义患者级表示，避免一名患者被当作多个独立个体。

### 5.2 输入与产出

输入为数值矩阵、$K$、初始化和停止标准。输出为硬标签、质心、每点到质心距离和 WCSS。

## 6. 适用场景

- 大样本、数值特征、簇近似球形且希望快速得到原型。
- 作为无监督分析基线或数据压缩工具。
- 不适合噪声多、任意形状、不同密度或要求软归属的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

model = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=50,
    random_state=42,
)
cluster = model.fit_predict(X_s)
centers_original = scaler.inverse_transform(model.cluster_centers_)
print("WCSS:", model.inertia_)
print("silhouette:", silhouette_score(X_s, cluster))
print(centers_original)
```

### 7.2 R

```r
set.seed(42)
x <- scale(df[, c("BMI", "TG", "HDL", "FastingGlucose", "SBP")])

fit <- kmeans(
  x,
  centers = 3,
  nstart = 50,
  iter.max = 100
)

cluster <- fit$cluster
fit$centers
fit$tot.withinss
```

## 8. 结果如何解释

- 质心是标准化空间中的平均原型，必要时变回原单位描述。
- 簇编号没有顺序含义，也可能在不同运行中交换。
- WCSS 只能用于同一数据预处理下比较；随 $K$ 增大必然下降。
- 聚类标签是探索性结构，不是诊断真值或因果分组。

## 9. 诊断与稳健性

1. 比较 $K$ 的肘部、silhouette 和稳定性。
2. 使用多次初始化并记录最佳与次佳 WCSS。
3. bootstrap 后比较样本共同聚类概率。
4. 改变标准化、变量集和异常值处理方案。
5. 在独立队列用最近质心分配后验证临床特征是否复现。

## 10. 推荐可视化

- WCSS 肘部图和 silhouette 随 $K$ 变化图。
- PCA 二维投影上的标签与质心。
- 标准化簇中心热图。
- 重采样共识矩阵。

## 11. 优势、局限与常见坑

**优势：** 简单、快速、可扩展，质心原型便于描述。

**局限：** 需指定 $K$，对尺度、初始化和离群点敏感，只适合特定簇形状。

**常见坑：** 未标准化；机械依赖肘部法；只跑一次；把类别编码直接纳入均值；把任意分组命名为疾病亚型。

## 12. 与相近方法的区别

- [[高斯混合模型（Gaussian Mixture Model, GMM）]]：GMM 给软概率并允许椭圆协方差。
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]：输出隶属度而非硬标签。
- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]：无需指定簇数，可标记噪声和非球形簇。
- 选择经验：簇紧凑、规模大、需要快速原型时先用 K-means。

## 13. 医学研究中的典型应用

- 代谢、炎症或实验室指标患者分群。
- 影像组学和组学降维后的探索性亚型。
- 医疗服务利用模式与患者画像原型。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| centroid | 簇内样本的均值向量 |
| inertia/WCSS | 所有点到所属质心的平方距离和 |
| Lloyd algorithm | 交替分配与更新质心的经典算法 |
| k-means++ | 让初始质心彼此分散的初始化方法 |
| hard clustering | 每个样本只属于一个簇 |

## 15. 相关方法

- [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]
- [[层次聚类（Hierarchical Clustering）]]

## 16. 参考资料

- MacQueen J. Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium*. 1967;1:281-297.
- Lloyd S. Least squares quantization in PCM. *IEEE Trans Inf Theory*. 1982;28(2):129-137.
- scikit-learn Developers. `KMeans` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) （访问日期：2026-07-09）

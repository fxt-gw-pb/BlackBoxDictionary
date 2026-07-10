---
title: Isomap
english_name: Isometric Mapping
slug: isometric-mapping
aliases: [Isomap, isometric mapping, "Isomap（Isometric Mapping）"]
category: 降维与表征学习
subcategory: 非线性流形学习
tags: [医学统计, 数据科学, 降维, 流形学习, 无监督学习]
status: 已建
difficulty: advanced
question_type: 流形测地距离保持降维
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [vegan, RDRToolbox]
---

# Isomap（Isometric Mapping）

## 1. 方法概览

### 1.1 一句话本质

Isomap 用近邻图上的最短路径近似流形测地距离，再用经典 MDS 把这些距离还原为低维坐标。

### 1.2 定义

Isomap 是一种非线性流形学习方法。它把欧氏距离的使用限制在局部近邻内，通过全局最短路径连接局部距离，目标是在低维中保持沿流形行走的距离。

### 1.3 它主要解决什么问题

- 高维样本沿弯曲曲面分布，直线距离会穿过曲面而低估真实间隔。
- 希望展开流形并兼顾较大尺度的几何关系。
- 需要从高维影像、组学或表型数据获得探索性坐标。

### 1.4 直觉与类比

地图上隔山相望的两家医院直线距离很近，但实际道路很远。Isomap 不穿山测直线，而是把局部可通行道路连成网络，再按最短路绘制一张低维地图。

## 2. 核心思想与原理

### 2.1 根本困难

原空间欧氏距离只在流形的很小邻域内近似真实距离。直接对全局欧氏距离做 MDS，容易把弯曲结构压扁或折叠。

### 2.2 关键洞察

如果样本沿流形采样足够密，近邻之间的短线段可近似局部测地线；把这些短线段的最短路径相加，就能近似远距离样本之间的流形距离。

### 2.3 三步算法

1. 建立 $k$ 近邻图或半径近邻图，以局部距离为边权。
2. 用 Dijkstra 或 Floyd-Warshall 算法求所有点对最短路径。
3. 对测地距离矩阵执行经典 MDS，得到低维坐标。

## 3. 数学形式

### 3.1 近邻图

对局部相邻的 $i,j$ 设置：

$$
G_{ij}=d(x_i,x_j)
$$

非邻接点暂不直接连边。

### 3.2 测地距离近似

图上最短路径距离为：

$$
d_G(i,j)=
\min_{\pi:i\rightsquigarrow j}
\sum_{(u,v)\in\pi}G_{uv}
$$

这个式子在所有连接 $i$ 与 $j$ 的路径中选择总边长最小的一条。

### 3.3 经典 MDS

令 $D_G^{(2)}$ 为 $d_G(i,j)^2$ 组成的矩阵，$H=I-\mathbf1\mathbf1^\top/n$：

$$
B=-\frac12HD_G^{(2)}H
$$

若 $B=V\Lambda V^\top$，取前 $q$ 个正特征值：

$$
Y=V_q\Lambda_q^{1/2}
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 样本密集覆盖流形 | 最短路径绕行或断裂 | 第 $k$ 近邻距离分布 |
| 近邻图连通 | 无穷测地距离 | 连通分量 |
| 图没有跨折叠捷径 | 低估测地距离 | 检查可疑长边 |
| 流形近似等距嵌入 | MDS 残差较大 | residual variance |
| 距离度量符合问题 | 图结构无医学意义 | 多种距离敏感性分析 |

## 4. 手把手算例

三个点形成一条弯折路径：

$$
A=(0,0),\qquad B=(1,0),\qquad C=(1,1)
$$

设近邻图只保留 $A-B$ 和 $B-C$。

**Step 1：比较直线距离。**

$$
d_E(A,C)=\sqrt{(1-0)^2+(1-0)^2}=\sqrt2
$$

直线会斜穿弯折内部。

**Step 2：计算图上测地距离。**

$$
d_G(A,B)=1,\qquad d_G(B,C)=1
$$

$$
d_G(A,C)=d_G(A,B)+d_G(B,C)=2
$$

因此测地距离矩阵为：

$$
D_G=
\begin{pmatrix}
0&1&2\\
1&0&1\\
2&1&0
\end{pmatrix}
$$

**Step 3：经典 MDS 展开。**

双中心化得到：

$$
B=
\begin{pmatrix}
1&0&-1\\
0&0&0\\
-1&0&1
\end{pmatrix}
$$

唯一正特征值为 2，对应坐标可取：

$$
Y=(-1,0,1)^\top
$$

低维距离重新变成 1、1、2。也就是说，Isomap 把二维弯折路径展开为一条直线，同时保留沿路径行走的距离。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 输入为样本乘特征矩阵，或可定义成对距离的数据。
- 连续特征通常需要标准化；高维数据可先 PCA 去噪。
- 缺失值需先处理；删失结局不是 Isomap 的直接输入。
- 对混合变量、序列或组学计数应选择有依据的距离，而非机械使用欧氏距离。

### 5.2 示例表格

| Patient | Imaging_1 | Imaging_2 | Imaging_3 | Outcome |
| --- | ---: | ---: | ---: | --- |
| P01 | 0.1 | 1.4 | 2.1 | stable |
| P02 | 0.4 | 1.2 | 1.8 | stable |
| P03 | 1.3 | 0.7 | 0.9 | progress |
| P04 | 1.8 | 0.2 | 0.3 | progress |

`Outcome` 可用于嵌入后验证，不应在无监督拟合前决定展示参数。

### 5.3 输入与产出

输入包括特征、近邻规则、距离度量和目标维数。输出包括嵌入坐标、近邻图、测地距离矩阵及重构误差；内存和计算通常随样本对数量快速增长。

## 6. 适用场景

- 连续、低维、近似等距且采样充分的弯曲流形。
- 影像姿态、连续生理状态或疾病演进轨迹探索。
- 不适合流形有分支、孔洞被稀疏采样、噪声很高或样本规模极大的场景。
- 若只关注局部可视化而不要求全局距离，UMAP 或 t-SNE 往往更实用。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.manifold import Isomap, trustworthiness
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = df[["Imaging_1", "Imaging_2", "Imaging_3"]]

model = make_pipeline(
    StandardScaler(),
    Isomap(
        n_neighbors=10,
        n_components=2,
        metric="euclidean",
        path_method="auto",
        n_jobs=-1,
    ),
)
embedding = model.fit_transform(X)

iso = model.named_steps["isomap"]
X_scaled = model.named_steps["standardscaler"].transform(X)
score = trustworthiness(X_scaled, embedding, n_neighbors=10)
result = pd.DataFrame(embedding, columns=["ISO1", "ISO2"])
print(iso.reconstruction_error(), score)
```

### 7.2 R

```r
library(vegan)

x <- scale(df[, c("Imaging_1", "Imaging_2", "Imaging_3")])
d <- dist(x, method = "euclidean")

fit <- isomap(
  d,
  ndim = 2,
  k = 10
)
embedding <- scores(fit, choices = 1:2)
head(embedding)
```

## 8. 结果如何解读

- 点间距离用于近似沿流形的关系，坐标轴方向和正负没有固定医学意义。
- 远距离是否可信取决于整条最短路径，不应只凭二维图判断。
- 图上桥接或分离可能来自 $k$ 的选择与采样空洞。
- 颜色分组应由未参与参数挑选的外部标签验证。

## 9. 假设诊断与稳健性

1. 检查近邻图是否连通，以及是否存在异常长边。
2. 画第 $k$ 近邻距离和局部密度分布。
3. 比较多组 $k$ 或半径，不仅挑最好看的结果。
4. 报告 reconstruction error、trustworthiness 与 residual variance。
5. 比较欧氏、余弦或领域特定距离。
6. bootstrap 后用 Procrustes 对齐嵌入，检查结构稳定性。
7. 下游模型在每个训练折内重新拟合标准化、PCA 与 Isomap。

## 10. 推荐可视化

- Isomap 二维嵌入并叠加近邻图。
- 测地距离与嵌入距离散点图。
- 重构误差随维数或近邻数的曲线。
- 不同 $k$ 的小多图与连通分量标记。

## 11. 优势、局限与常见坑

**优势：** 原理清晰；能展开某些弯曲流形；比只保留局部邻域的方法更强调全局测地结构。

**局限：** 易受短路边、断图和稀疏采样影响；成对最短路径成本高；只适合近似等距结构。

**常见坑：** $k$ 过大造成跨折叠捷径；$k$ 过小使图断开；忽视特征量纲；把二维直线距离当原始欧氏距离；只按标签分离度选参数。

## 12. 与相近方法的区别

- [[多维尺度分析（Multidimensional Scaling, MDS）]]：MDS 直接使用给定不相似度；Isomap 先用近邻图估计测地距离。
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]：LLE 保留局部重构权重，不直接保持全局测地距离。
- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 保持全局线性方差；Isomap 可展开非线性流形。
- [[UMAP（Uniform Manifold Approximation and Projection）]]：UMAP 优化模糊邻域图，通常更快，但其二维距离不等同于测地距离。
- **如何选择：** 有充分理由相信等距流形且重视沿流形距离时用 Isomap；只需稳定线性基线时先用 PCA；强调局部可视化时考虑 UMAP。

## 13. 医学研究中的典型应用

- 影像姿态、器官形态或病灶形变的连续表征。
- 多指标疾病进展路径探索。
- 高维生理信号状态空间可视化。

研究报告应同时给出预处理、距离、近邻规则、连通性和敏感性分析。

## 14. 关键术语

| 术语 | 含义 |
| --- | --- |
| geodesic distance 测地距离 | 沿流形表面可行路径的最短长度 |
| neighborhood graph 近邻图 | 以样本为节点、局部近邻为边的图 |
| shortest path 最短路径 | 图上连接两点且总边权最小的路径 |
| isometry 等距映射 | 尽可能保持距离的映射 |
| short-circuit edge 短路边 | 跨越流形折叠、错误缩短测地距离的边 |
| residual variance 残余方差 | 测地距离与嵌入距离不一致程度的指标 |

## 15. 相关方法

- [[多维尺度分析（Multidimensional Scaling, MDS）]]
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]

## 16. 参考资料

- Tenenbaum JB, de Silva V, Langford JC. A global geometric framework for nonlinear dimensionality reduction. *Science*. 2000;290(5500):2319-2323.
- de Silva V, Tenenbaum JB. Global versus local methods in nonlinear dimensionality reduction. *Advances in Neural Information Processing Systems*. 2003;15.
- scikit-learn Developers. `Isomap` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html) （访问日期：2026-07-09）

---
title: UMAP
english_name: Uniform Manifold Approximation and Projection
slug: uniform-manifold-approximation-and-projection
aliases: [UMAP, "UMAP（Uniform Manifold Approximation and Projection）"]
category: 降维与表征学习
subcategory: 非线性流形学习
tags: [医学统计, 数据科学, 降维, 可视化, 流形学习]
status: 已建
difficulty: advanced
question_type: 高维邻域结构与拓扑表征可视化
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [umap-learn, scikit-learn]
r_packages: [uwot]
---

# UMAP（Uniform Manifold Approximation and Projection）

## 1. 方法概览

### 1.1 一句话本质

UMAP 把高维近邻关系做成带置信度的模糊图，再优化低维点的位置，使高置信边靠近、缺失边保持分离。

### 1.2 定义

UMAP 是一种非线性流形学习与可视化方法。它在局部尺度校正后构造模糊单纯复形的一维骨架，可直观理解为加权近邻图，再最小化高维图和低维图之间的模糊集合交叉熵。

### 1.3 它主要解决什么问题

- 高维样本的局部邻域难以直接展示。
- 不同区域密度不同，固定距离尺度难以同时适配。
- 需要较快地处理大规模数据，并可能把新样本映射到已有嵌入。

### 1.4 直觉与类比

每个样本都按自己所在街区的尺度描述“我们有多像邻居”，再把双向看法合并为一张关系网。低维排座时，强关系频繁拉近，随机抽到的非邻居负责推开。

## 2. 核心思想与原理

### 2.1 根本困难

高密度区域和低密度区域不能共用同一个距离阈值；同时，直接保持所有成对距离既昂贵，也可能牺牲真正关心的局部结构。

### 2.2 关键洞察

- 每个样本用局部连通距离 $\rho_i$ 和尺度 $\sigma_i$ 校准自己的邻域。
- 将方向性邻域通过模糊并集合并，形成对称边权。
- 在低维用可微重尾曲线表示边概率，并以吸引和排斥共同优化。

### 2.3 参数控制的尺度

`n_neighbors` 决定构图时观察多宽的邻域，`min_dist` 控制低维中相似点可以压得多紧。它们改变的是展示尺度，不是从数据中自动检验出的真值。

## 3. 数学形式

### 3.1 局部方向性边权

对 $j\in\mathcal N_i$：

$$
p_{j\mid i}=
\exp\left[
-\frac{\max(0,d(x_i,x_j)-\rho_i)}{\sigma_i}
\right]
$$

其中 $\rho_i$ 保障局部连通，$\sigma_i$ 适应局部密度。

### 3.2 模糊并集

双向关系合并为：

$$
p_{ij}
=p_{j\mid i}+p_{i\mid j}
-p_{j\mid i}p_{i\mid j}
$$

只要任一方向认为关系较强，合并后的边权就不会很低。

### 3.3 低维边权与目标

$$
q_{ij}=
\frac1{1+a\|y_i-y_j\|^{2b}}
$$

一对样本的交叉熵贡献可写为：

$$
-p_{ij}\log q_{ij}
-(1-p_{ij})\log(1-q_{ij})
$$

$a,b$ 由 `min_dist` 与 `spread` 拟合。实现中通常通过边采样和负采样近似优化，而非枚举所有样本对。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 局部距离有意义 | 模糊图无领域意义 | 预处理与距离敏感性 |
| 近邻图大体连通 | 出现任意分离的岛 | 连通分量与断边警告 |
| 采样能覆盖结构 | 稀疏区被拉伸或切断 | 邻域距离分布 |
| 参数结论稳定 | 视觉叙事依赖单一设定 | 参数网格与多种种子 |
| 标签未用于挑选无监督图 | 产生确认偏倚 | 预先规定分析流程 |

## 4. 手把手算例

设样本 $i$ 到近邻 $j$ 的距离为 2，局部连通距离 $\rho_i=1$，局部尺度 $\sigma_i=1$。

**Step 1：计算 $i$ 看 $j$ 的方向性权重。**

$$
p_{j\mid i}
=\exp[-(2-1)/1]
=e^{-1}
\approx0.368
$$

假设反方向因局部密度不同，有：

$$
p_{i\mid j}=0.600
$$

**Step 2：取模糊并集。**

$$
p_{ij}
=0.368+0.600-(0.368)(0.600)
\approx0.747
$$

合并后，这是一条较强的邻边。

**Step 3：检查低维布局。** 为便于手算，令 $a=b=1$，低维距离为 0.5：

$$
q_{ij}
=\frac1{1+0.5^2}
=0.800
$$

该边的交叉熵贡献为：

$$
-0.747\log(0.800)
-0.253\log(0.200)
\approx0.574
$$

优化会同时考虑许多强边的吸引与负样本的排斥。这里的数字说明：局部尺度不同的双向判断可合成一个目标边权，再由低维距离去逼近它。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 样本乘特征矩阵、稀疏矩阵或预计算距离。
- 高维连续数据常标准化后先 PCA 或 SVD 去噪。
- 单细胞计数通常先完成领域适当的归一化和高变特征选择。
- 缺失值需先处理；删失结局不是无监督 UMAP 的直接输入。
- 重复测量数据需防止个体身份或时间批次主导邻域。

### 5.2 示例表格

| Patient | PC1 | PC2 | PC3 | Subtype |
| --- | ---: | ---: | ---: | --- |
| P01 | 1.8 | -0.4 | 0.2 | A |
| P02 | 1.6 | -0.3 | 0.4 | A |
| P03 | -1.1 | 1.2 | -0.2 | B |
| P04 | -1.4 | 1.0 | -0.1 | B |

`Subtype` 只用于无监督嵌入后的外部核对。

### 5.3 输入与产出

输入包括特征、距离、`n_neighbors`、`min_dist`、目标维数与随机种子。输出包括低维坐标、模糊近邻图和可选的新样本变换模型；不确定性需用参数、种子和重采样稳定性表达。

## 6. 适用场景

- 大规模高维数据的局部结构展示和候选亚群探索。
- 单细胞、影像组学、多组学及深度嵌入。
- 需要在固定预处理流程下近似映射新样本的场景。
- 不适合把二维距离当效应量、把图形当显著性检验或要求坐标轴直接解释。
- 若保留密度本身很重要，可评估 densMAP，而不是假设普通 UMAP 保持密度。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

features = [col for col in df.columns if col.startswith("Feature_")]
X_scaled = StandardScaler().fit_transform(df[features])
X_pca = PCA(
    n_components=30,
    svd_solver="randomized",
    random_state=42,
).fit_transform(X_scaled)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42,
    transform_seed=42,
)
embedding = reducer.fit_transform(X_pca)

score = trustworthiness(X_pca, embedding, n_neighbors=15)
result = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
print(score, reducer.graph_.shape)
```

### 7.2 R

```r
library(uwot)

x <- scale(df[, grep("^Feature_", names(df))])
pca <- prcomp(x, center = FALSE, scale. = FALSE)
x_pca <- pca$x[, 1:30, drop = FALSE]

set.seed(42)
embedding <- umap(
  x_pca,
  n_neighbors = 15,
  min_dist = 0.1,
  n_components = 2,
  metric = "euclidean",
  ret_model = FALSE,
  n_threads = 1
)
head(embedding)
```

## 8. 结果如何解读

- 解释局部邻接、连续梯度及在多组参数下重复出现的结构。
- UMAP1、UMAP2 的方向、尺度和单位没有直接医学含义。
- 簇间距离、面积和密度可能被优化过程改变，不能直接作定量比较。
- 图上的岛可能来自真实分群，也可能来自构图断裂、批次或预处理。

## 9. 假设诊断与稳健性

1. 检查模糊图的连通分量、孤立点及断边警告。
2. 系统比较 `n_neighbors`、`min_dist`、距离和随机种子。
3. 报告 trustworthiness、近邻重叠率及必要时 continuity。
4. 分别按批次、中心、测序深度、缺失率和质量指标着色。
5. bootstrap 或子采样后对嵌入做对齐与邻域比较。
6. 新样本投影需检查其到训练分布的距离，避免外推。
7. 监督建模时，标准化、PCA 和 UMAP 都必须在训练折内拟合。

## 10. 推荐可视化

- UMAP 散点图，分别按生物标签和技术因素着色。
- `n_neighbors` 与 `min_dist` 参数网格。
- 原空间和低维空间的近邻保持图。
- 模糊近邻图连通分量与边权分布。

## 11. 优势、局限与常见坑

**优势：** 扩展性较好；局部尺度自适应；通常支持新样本变换；距离和监督变体较灵活。

**局限：** 参数与随机性会改变布局；普通 UMAP 不保证全局距离或密度；理论假设不等同于数据必然满足流形模型。

**常见坑：** 根据标签挑最分离的图；把岛直接命名为新亚型；忽略批次和质量因素；全数据拟合造成泄漏；对训练分布之外样本盲目 `transform`。

## 12. 与相近方法的区别

- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]：t-SNE 匹配邻居概率；UMAP 匹配模糊图，通常更快且可变换新样本。
- [[Isomap（Isometric Mapping）]]：Isomap 保持最短路径测地距离；UMAP 不把远距离作为精确尺度。
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]：LLE 保留局部重构权重；UMAP 用带权近邻边。
- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 是线性方差投影，可解释性和可重复性更强。
- **如何选择：** 大规模局部可视化与新样本映射可优先评估 UMAP；需严格线性基线和载荷解释时用 PCA；所有视觉发现都需独立验证。

## 13. 医学研究中的典型应用

- 单细胞类型、状态和发育连续谱展示。
- 多组学患者表征及候选亚群探索。
- 影像组学或临床高维特征的样本相似性可视化。

研究报告至少应给出特征处理、PCA 维数、距离、`n_neighbors`、`min_dist`、随机种子和稳定性分析。

## 14. 关键术语

| 术语 | 含义 |
| --- | --- |
| fuzzy simplicial set 模糊单纯复形 | 用零到一的成员强度表示局部拓扑关系 |
| membership strength 成员强度 | 一条邻边属于局部结构的置信度 |
| local connectivity 局部连通性 | 确保每个样本至少与最近邻强连接的设定 |
| fuzzy union 模糊并集 | 合并双向邻域置信度的运算 |
| negative sampling 负采样 | 抽取非邻边近似排斥项以加速优化 |
| transform 新样本变换 | 在已拟合嵌入基础上定位新观测 |

## 15. 相关方法

- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[Isomap（Isometric Mapping）]]
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426; 2018.
- McInnes L, Healy J, Saul N, Grossberger L. UMAP: Uniform Manifold Approximation and Projection. *Journal of Open Source Software*. 2018;3(29):861.
- umap-learn Developers. `UMAP` API Guide. [https://umap-learn.readthedocs.io/en/latest/api.html](https://umap-learn.readthedocs.io/en/latest/api.html) （访问日期：2026-07-09）

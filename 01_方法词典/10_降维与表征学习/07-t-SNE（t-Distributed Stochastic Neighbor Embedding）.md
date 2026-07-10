---
title: t-SNE
english_name: t-Distributed Stochastic Neighbor Embedding
slug: t-distributed-stochastic-neighbor-embedding
aliases: [t-SNE, TSNE, "t-SNE（t-Distributed Stochastic Neighbor Embedding）"]
category: 降维与表征学习
subcategory: 非线性可视化
tags: [医学统计, 数据科学, 降维, 可视化, 流形学习]
status: 已建
difficulty: advanced
question_type: 高维样本邻域可视化
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn, openTSNE]
r_packages: [Rtsne]
---

# t-SNE（t-Distributed Stochastic Neighbor Embedding）

## 1. 方法概览

### 1.1 一句话本质

t-SNE 把高维和低维的邻近关系都转成概率，并移动低维点，使真正的高维近邻在图上仍有较高相遇概率。

### 1.2 定义

t-SNE 是面向二维或三维可视化的非线性嵌入方法。高维相似度使用自适应带宽的高斯核，低维相似度使用重尾 Student $t$ 分布，再最小化二者的 Kullback-Leibler 散度。

### 1.3 它主要解决什么问题

- 高维局部邻域无法直接观察。
- PCA 的线性投影掩盖弯曲或复杂的局部结构。
- 需要探索单细胞、组学、影像或深度特征中的潜在亚群。

### 1.4 直觉与类比

先为每个样本列一张“最可能成为邻居”的名单，再在二维会场安排座位，让名单上的人尽量坐近。名单外的距离只提供较弱约束，所以远处簇间空白不能按比例解释。

## 2. 核心思想与原理

### 2.1 根本困难

高维空间容易出现距离集中，低维空间又容纳不下所有邻近关系，造成拥挤问题：很多中等距离点会被挤到中心。

### 2.2 关键洞察

- 用每个点自己的高斯带宽，把邻域尺度调到相近的有效邻居数。
- 低维使用重尾 $t$ 分布，使非近邻能被推得更远，为真正近邻腾出空间。
- 用非对称的 KL 目标强烈惩罚“高维近邻在图上被拆开”。

### 2.3 优化而非唯一解

t-SNE 目标非凸，初始化、随机种子与超参数都会影响布局。它给出的是局部结构的一种可视化，不是唯一坐标系，也不是聚类检验。

## 3. 数学形式

### 3.1 高维条件相似度

$$
p_{j\mid i}=
\frac{
\exp\left(-\|x_i-x_j\|^2/(2\sigma_i^2)\right)
}{
\sum_{k\ne i}
\exp\left(-\|x_i-x_k\|^2/(2\sigma_i^2)\right)
}
$$

对称化后：

$$
p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}
$$

### 3.2 Perplexity

以二进制熵 $H(P_i)$ 定义：

$$
\operatorname{Perp}(P_i)=2^{H(P_i)}
$$

算法通过搜索 $\sigma_i$ 使每个点达到指定 perplexity。它可粗略理解为关注的有效邻居数，但并不等于固定的 $k$。

### 3.3 低维相似度与目标

$$
q_{ij}=
\frac{(1+\|y_i-y_j\|^2)^{-1}}
{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}}
$$

$$
C=\operatorname{KL}(P\|Q)
=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}
$$

若某对样本在高维有较大 $p_{ij}$，却在低维有很小 $q_{ij}$，损失会明显增大。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 距离能表达样本相似性 | 邻域无领域意义 | 预处理与距离比较 |
| 样本量显著大于 perplexity | 带宽搜索失真 | 保证 perplexity 小于样本数 |
| 优化达到稳定解 | 图形随种子剧变 | 多种种子与 KL 轨迹 |
| 局部结构是主要目标 | 误读簇间距离和大小 | 与原空间邻域核对 |
| 参数未按标签挑图 | 产生确认偏倚 | 预先规定参数网格 |

## 4. 手把手算例

固定中心样本 $i$，它到两个候选邻居 $j,k$ 的高维距离分别为 1 和 2，令 $\sigma_i=1$。

**Step 1：计算未归一化高维权重。**

$$
s_{ij}=e^{-1^2/2}\approx0.6065
$$

$$
s_{ik}=e^{-2^2/2}=e^{-2}\approx0.1353
$$

**Step 2：归一化。**

$$
p_{j\mid i}
=\frac{0.6065}{0.6065+0.1353}
\approx0.8176
$$

$$
p_{k\mid i}\approx0.1824
$$

因此 $j$ 是更强的局部邻居。

**Step 3：检查一个低维候选布局。** 若低维距离为 1 和 3，则 Student $t$ 权重分别为：

$$
r_{ij}=\frac1{1+1^2}=0.5,
\qquad
r_{ik}=\frac1{1+3^2}=0.1
$$

在这两个候选中归一化：

$$
q_{j\mid i}=0.8333,
\qquad
q_{k\mid i}=0.1667
$$

其局部 KL 贡献约为：

$$
0.8176\log\frac{0.8176}{0.8333}
+0.1824\log\frac{0.1824}{0.1667}
\approx0.0009
$$

这个候选布局很好地复现了中心点的邻居概率。完整 t-SNE 会对所有点对使用对称概率并共同优化坐标。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 输入为样本乘特征矩阵，或预计算距离矩阵。
- 高维稠密数据通常先标准化并降到约 30 至 50 个主成分。
- 稀疏数据可先用 Truncated SVD。
- 缺失值需先处理；删失时间、类别标签不是无监督 t-SNE 的直接输入。
- 重复测量应考虑个体内相关，避免把同一患者多次记录形成的簇误认为疾病亚型。

### 5.2 示例表格

| Cell | PC1 | PC2 | PC3 | Cell_type |
| --- | ---: | ---: | ---: | --- |
| C01 | 2.1 | -0.4 | 0.8 | T |
| C02 | 1.9 | -0.2 | 0.6 | T |
| C03 | -1.2 | 1.4 | -0.3 | B |
| C04 | -1.4 | 1.1 | -0.5 | B |

标签只用于嵌入后着色和外部验证。

### 5.3 输入与产出

输入包括特征或距离、perplexity、学习率、迭代次数、初始化与随机种子。输出主要是低维坐标和最终 KL divergence；标准 scikit-learn `TSNE` 不提供通用的训练后新样本 `transform`。

## 6. 适用场景

- 适合高维样本局部邻域和潜在亚群的探索性可视化。
- 常用于单细胞、空间组学、影像表征和神经网络嵌入。
- 不适合解释坐标轴、比较簇间绝对距离、估计簇大小或直接完成统计推断。
- 若需要稳定投影新样本，可考虑参数化方法、openTSNE 的变换流程或 UMAP，并单独验证。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

features = [col for col in df.columns if col.startswith("Gene_")]
X_scaled = StandardScaler().fit_transform(df[features])
X_pca = PCA(
    n_components=30,
    svd_solver="randomized",
    random_state=42,
).fit_transform(X_scaled)

model = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42,
)
embedding = model.fit_transform(X_pca)

score = trustworthiness(X_pca, embedding, n_neighbors=15)
result = pd.DataFrame(embedding, columns=["TSNE1", "TSNE2"])
print(model.kl_divergence_, score)
```

### 7.2 R

```r
library(Rtsne)

x <- scale(df[, grep("^Gene_", names(df))])
pca <- prcomp(x, center = FALSE, scale. = FALSE)
x_pca <- pca$x[, 1:30, drop = FALSE]

set.seed(42)
fit <- Rtsne(
  x_pca,
  dims = 2,
  perplexity = 30,
  theta = 0.5,
  pca = FALSE,
  max_iter = 1000,
  check_duplicates = TRUE
)
embedding <- fit$Y
fit$itercosts
```

## 8. 结果如何解读

- 优先解释局部邻近和重复出现的结构。
- t-SNE1、t-SNE2 的方向、尺度和单位没有直接医学含义。
- 簇间空白、相对方向和表面积通常不代表原空间的定量关系。
- 图上有簇不等于存在可重复亚型，应回到原始变量和独立队列验证。

## 9. 假设诊断与稳健性

1. 用多组 perplexity、种子和初始化重复拟合。
2. 报告最终 KL divergence、迭代次数和 trustworthiness。
3. 比较 PCA 预降维维数及标准化方案。
4. 计算原空间与嵌入空间的近邻重叠率。
5. 检查批次、测序深度、中心和缺失模式是否主导布局。
6. 对重复运行用 Procrustes 或邻域指标比较，不直接比坐标。
7. 若用于下游建模，所有预处理必须在训练折内部完成。

## 10. 推荐可视化

- t-SNE 散点图，分别按生物标签、批次与质量指标着色。
- perplexity 和随机种子敏感性小多图。
- KL divergence 或优化成本随迭代变化图。
- 原空间与低维空间近邻重叠率图。

## 11. 优势、局限与常见坑

**优势：** 局部结构展示能力强；重尾分布缓解拥挤；成熟实现可处理较大数据。

**局限：** 非凸、随机且参数敏感；全局几何不可靠；标准实现难以自然投影新样本；二维图容易被过度解读。

**常见坑：** 把簇间距离当效应大小；按标签挑最分离的参数；不报告随机种子；跳过批次诊断；直接在 t-SNE 坐标上做正式聚类或差异检验。

## 12. 与相近方法的区别

- [[UMAP（Uniform Manifold Approximation and Projection）]]：两者都强调邻域；UMAP 使用模糊图目标，通常更快并支持拟合后变换。
- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 是确定性的线性方差投影，坐标和全局关系更易解释。
- [[Isomap（Isometric Mapping）]]：Isomap 试图保持测地距离；t-SNE 主要保持概率邻域。
- **如何选择：** 局部探索性可视化可用 t-SNE；需要新样本映射或较大规模时优先评估 UMAP；任何非线性图都应与 PCA 基线并列。

## 13. 医学研究中的典型应用

- 单细胞转录组细胞类型和状态展示。
- 病理图像或放射影像深度特征可视化。
- 多组学患者相似性与候选亚群探索。

报告中应明确预处理、PCA 维数、perplexity、学习率、初始化、随机种子和重复运行策略。

## 14. 关键术语

| 术语 | 含义 |
| --- | --- |
| stochastic neighbor embedding 随机邻域嵌入 | 用邻居概率表达并匹配高低维关系 |
| perplexity 困惑度 | 由邻域概率熵得到的有效邻域尺度 |
| Student t distribution Student t 分布 | 用于低维相似度的重尾分布 |
| crowding problem 拥挤问题 | 高维邻域无法无失真塞入低维的现象 |
| KL divergence KL 散度 | 衡量两组概率分布失配的非对称量 |
| early exaggeration 早期夸大 | 优化初期放大高维吸引关系的策略 |

## 15. 相关方法

- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Isomap（Isometric Mapping）]]
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- van der Maaten L, Hinton G. Visualizing data using t-SNE. *Journal of Machine Learning Research*. 2008;9:2579-2605.
- van der Maaten L. Accelerating t-SNE using tree-based algorithms. *Journal of Machine Learning Research*. 2014;15:3221-3245.
- scikit-learn Developers. `TSNE` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) （访问日期：2026-07-09）

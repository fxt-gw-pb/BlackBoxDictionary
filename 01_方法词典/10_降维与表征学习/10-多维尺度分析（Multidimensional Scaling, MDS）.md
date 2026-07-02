---
title: 多维尺度分析
english_name: Multidimensional Scaling, MDS
slug: multidimensional-scaling-mds
aliases: [MDS, multidimensional scaling, classical MDS, metric MDS, "多维尺度分析（Multidimensional Scaling, MDS）"]
category: 降维与表征学习
subcategory: 距离保持降维
tags: [医学统计, 数据科学, 降维, 距离矩阵, 可视化]
status: 已建
difficulty: intermediate
question_type: 距离矩阵低维可视化
data_type: [距离矩阵, 相似度矩阵, 高维特征矩阵]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [stats, smacof]
---

# 多维尺度分析（Multidimensional Scaling, MDS）

## 1. 方法概览

### 1.1 定义

多维尺度分析是一类把样本间距离或不相似度表示为低维坐标的方法。它的目标是在二维或三维空间中尽量保留原始距离关系。

### 1.2 它主要解决什么问题

- 研究问题：只有样本间距离或相似度时，如何把对象放到低维空间中展示。
- 适用任务：距离矩阵可视化、相似性结构探索、样本关系展示。
- 常见医学场景：患者相似性网络可视化，微生物群落 beta 多样性展示，临床文本或表型相似度嵌入。

### 1.3 直觉理解

MDS 像是在一张地图上摆放样本点：原始距离近的样本在图上也应靠近，原始距离远的样本在图上也应远离。不同 MDS 版本对“距离保留”的定义不同。

## 2. 数学形式

### 2.1 核心公式

经典 MDS 从距离矩阵 $D=(d_{ij})$ 出发，对平方距离做双中心化：

$$
B=-\frac{1}{2}HD^{(2)}H
$$

其中：

$$
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^\top
$$

对 $B$ 做特征分解：

$$
B=V\Lambda V^\top
$$

取前 $k$ 个正特征值对应的坐标：

$$
Y=V_k\Lambda_k^{1/2}
$$

度量 MDS 常最小化 stress：

$$
\operatorname{Stress}(Y)=
\sqrt{
\frac{\sum_{i<j}(d_{ij}-\|y_i-y_j\|)^2}
{\sum_{i<j}d_{ij}^2}
}
$$

### 2.2 参数或统计量含义

- $D$：原始距离或不相似度矩阵。
- $B$：由距离矩阵恢复的内积矩阵。
- $Y$：低维坐标。
- `n_components`：目标低维维度。
- stress：低维距离与原始距离不一致的程度。
- metric/nonmetric：是否保留距离数值，还是只保留距离排序。

### 2.3 关键假设

- 距离或不相似度能合理表达样本关系。
- 低维空间足以近似原始距离结构。
- 经典 MDS 对欧氏距离结构最自然，非欧氏距离可能出现负特征值。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：特征矩阵、距离矩阵或相似度转换后的不相似度矩阵。
- 因变量类型：MDS 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵，或样本乘以样本距离矩阵。
- 是否适合高维数据：适合基于距离降维，但大样本距离矩阵成本较高。
- 是否适合缺失较多数据：需先能计算可靠距离。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需定义合适的样本距离，普通 MDS 不建模相关性。

### 3.2 示例表格

以患者间距离矩阵为例：

| Patient | P001 | P002 | P003 | P004 |
| --- | --- | --- | --- | --- |
| P001 | 0.00 | 0.42 | 1.10 | 0.95 |
| P002 | 0.42 | 0.00 | 1.04 | 0.88 |
| P003 | 1.10 | 1.04 | 0.00 | 0.36 |
| P004 | 0.95 | 0.88 | 0.36 | 0.00 |

### 3.3 输入与产出

#### 输入

- 输入数据：距离矩阵或可计算距离的特征矩阵。
- 关键变量：距离度量、MDS 类型、目标维度、随机种子。
- 需要预处理的内容：标准化、距离矩阵检查、缺失处理、异常样本检查。

#### 产出

- 模型对象/统计结果：低维坐标、stress、特征值或拟合优度。
- 参数估计：经典 MDS 的特征向量坐标，或迭代优化得到的坐标。
- 预测结果：不直接预测。
- 不确定性指标：stress、重采样稳定性、不同距离度量下嵌入一致性。

## 4. 适用场景

- 适合：核心输入是距离/相似度、希望展示样本关系、距离定义比原始变量更重要的场景。
- 不适合：需要解释原始变量贡献、样本量很大导致距离矩阵不可承受、低维保距效果很差的场景。
- 使用前需要特别检查的点：距离度量是否与研究问题匹配，stress 是否可接受，离群样本是否主导图形。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("patient_features.csv")
features = ["Age", "BMI", "CRP", "LDL"]
X = StandardScaler().fit_transform(df[features])

dist = pairwise_distances(X, metric="euclidean")
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    normalized_stress="auto",
    random_state=42
)
embedding = mds.fit_transform(dist)

embedding_df = pd.DataFrame(embedding, columns=["MDS1", "MDS2"])
print("Stress:", mds.stress_)
print(embedding_df.head())
```

### 5.2 R

常用包：

- `stats`
- `smacof`

```r
x <- scale(df[, c("Age", "BMI", "CRP", "LDL")])
d <- dist(x)

fit <- cmdscale(d, k = 2, eig = TRUE)
embedding <- fit$points
head(embedding)

# Metric MDS with stress diagnostics:
# library(smacof)
# fit2 <- mds(d, ndim = 2, type = "ratio")
# fit2$stress
```

## 6. 结果如何解释

- 核心结果看什么：二维坐标中的距离关系、stress 或拟合优度、不同临床分组是否在距离意义下接近或分离。
- 每个主要参数如何解释：`dissimilarity="precomputed"` 表示输入已经是样本间距离矩阵。
- 临床或医学意义如何表达：MDS 图反映的是所选距离度量下的样本关系，结论必须绑定距离定义。
- 常见误读：坐标轴通常没有直接医学含义，点的绝对方向和旋转也不重要。

## 7. 推荐可视化

- MDS1-MDS2 样本散点图。
- stress 随维度变化曲线。
- 原始距离与低维距离散点图。
- 不同距离度量下的 MDS 图对比。

## 8. 优势、局限与常见坑

### 优势

- 可直接处理距离或不相似度矩阵。
- 结果直观，适合展示样本关系。
- 经典 MDS 与距离几何关系清晰。

### 局限

- 大样本距离矩阵存储和计算成本高。
- 坐标轴解释有限。
- 距离度量选择会强烈影响结果。

### 常见坑

- 忽略 stress，直接解释保距效果很差的图。
- 混用不同量纲变量而不标准化。
- 把 MDS 图上的视觉分群当作正式聚类或分类证据。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 从特征矩阵的线性方差结构出发，MDS 可直接从距离矩阵出发。
- 和 [[Isomap（Isometric Mapping）]] 的区别：Isomap 先估计流形测地距离，再用经典 MDS；MDS 本身可用于任何合适距离矩阵。
- 和 [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]] 的区别：t-SNE 强调局部概率邻域，MDS 更直接地优化距离保留。

## 10. 医学研究中的典型应用

- 微生物组 beta 多样性距离的样本可视化。
- 患者相似性矩阵的二维展示。
- 临床文本、症状或表型不相似度的结构探索。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Isomap（Isometric Mapping）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 12. 参考资料

- Kruskal JB. Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika*. 1964;29:1-27.
- Cox TF, Cox MAA. *Multidimensional Scaling*. 2nd ed. Chapman and Hall/CRC; 2000.
- scikit-learn Developers. `sklearn.manifold.MDS`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) （访问日期：2026-07-02）

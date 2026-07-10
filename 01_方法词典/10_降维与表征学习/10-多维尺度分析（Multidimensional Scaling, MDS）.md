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

### 1.1 一句话本质

MDS 只根据对象间不相似度寻找低维坐标，使图上点距尽量复现原始距离或距离顺序。

### 1.2 定义

多维尺度分析是一类距离表示方法。经典 MDS 对平方欧氏距离双中心化并做特征分解；度量 MDS 直接最小化距离差异，非度量 MDS 主要保持不相似度排序。

### 1.3 它主要解决什么问题

- 只有患者间距离而没有可直接使用的坐标。
- 距离定义比原始变量更符合研究问题。
- 需要二维或三维展示对象相似性。

### 1.4 直觉与类比

已知城市两两路程，但没有地图坐标。MDS 尝试在纸上摆放城市，让纸面距离尽量符合路程；图形是否可信取决于原始距离和低维拟合误差。

## 2. 核心思想与原理

### 2.1 从距离恢复内积

欧氏点的平方距离可由内积矩阵计算。双中心化逆向恢复中心化 Gram 矩阵，再通过特征分解得到坐标。

### 2.2 经典、度量与非度量

- 经典 MDS：代数解，最自然地处理欧氏距离。
- 度量 MDS：迭代最小化原距离与嵌入距离差。
- 非度量 MDS：允许单调变换，只强调距离排序。

### 2.3 方向没有解释

嵌入整体平移、旋转或镜像都保持点间距离。因此 MDS1/MDS2 轴的正负和方向通常没有固定医学含义。

## 3. 数学形式

### 3.1 经典 MDS

设 $D^{(2)}$ 为逐元素平方距离矩阵：

$$
H=I-\frac1n\mathbf1\mathbf1^\top
$$

$$
B=-\frac12HD^{(2)}H
$$

### 3.2 坐标恢复

$$
B=V\Lambda V^\top
$$

取前 $k$ 个正特征值：

$$
Y=V_k\Lambda_k^{1/2}
$$

### 3.3 Stress

一个常见归一化 Stress-1 为：

$$
\operatorname{Stress}_1=
\sqrt{
\frac{\sum_{i\lt j}(d_{ij}-\widehat d_{ij})^2}
{\sum_{i\lt j}d_{ij}^2}
}
$$

其中 $\widehat d_{ij}=\|y_i-y_j\|$。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 不相似度定义合理 | 图表达错误关系 | 比较距离度量 |
| 低维可近似原结构 | stress 高、视觉误导 | stress 与 Shepard 图 |
| 经典 MDS 距离近似欧氏 | 出现负特征值 | 特征值谱 |
| 样本量可承受距离矩阵 | 二次存储过大 | 子采样或近似 |

## 4. 手把手算例

三个对象的距离为：

$$
D=
\begin{pmatrix}
0&1&2\\
1&0&1\\
2&1&0
\end{pmatrix}
$$

直觉上它们应排在一条直线上。

**Step 1：平方距离。**

$$
D^{(2)}=
\begin{pmatrix}
0&1&4\\
1&0&1\\
4&1&0
\end{pmatrix}
$$

**Step 2：双中心化。** 使用 $B=-HD^{(2)}H/2$ 得：

$$
B=
\begin{pmatrix}
1&0&-1\\
0&0&0\\
-1&0&1
\end{pmatrix}
$$

**Step 3：特征分解。** 唯一正特征值为 $\lambda_1=2$，对应单位特征向量可取：

$$
v_1=\frac1{\sqrt2}(-1,0,1)^\top
$$

于是坐标：

$$
Y=v_1\sqrt{\lambda_1}=(-1,0,1)^\top
$$

**Step 4：核对距离。** 新坐标距离为 1、1、2，与原矩阵完全一致，stress 为 0。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 对称、对角为 0 的不相似度矩阵，或可计算距离的特征矩阵。
- 相似度需先有依据地转为不相似度。
- 缺失距离需要专门方法，不能随意填 0。

### 5.2 输入与产出

输入为距离矩阵、MDS 类型、维数与优化设置。输出为低维坐标、stress、距离拟合和经典 MDS 特征值。

## 6. 适用场景

- 微生物组 beta 多样性、患者相似性和非欧氏表型距离。
- 需要直观展示距离关系。
- 不适合超大样本、距离定义不可靠或要求轴有直接变量解释的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
dist = pairwise_distances(X_s, metric="euclidean")

model = MDS(
    n_components=2,
    metric_mds=True,
    metric="precomputed",
    normalized_stress=True,
    n_init=8,
    max_iter=500,
    eps=1e-6,
    random_state=42,
    n_jobs=-1,
)
embedding = model.fit_transform(dist)
print(model.stress_, model.n_iter_)
```

### 7.2 R

```r
x <- scale(df[, c("Age", "BMI", "CRP", "LDL")])
d <- dist(x, method = "euclidean")

classical <- cmdscale(
  d,
  k = 2,
  eig = TRUE,
  add = FALSE
)
embedding <- classical$points
classical$eig

# library(smacof)
# metric_fit <- mds(d, ndim = 2, type = "ratio")
# metric_fit$stress
```

## 8. 结果如何解释

- 点间相对距离比坐标轴方向更重要。
- stress 越低，低维距离越接近目标不相似度；必须注明 stress 定义。
- 经典 MDS 的负特征值提示距离并非完全欧氏。
- 图上视觉分群不能替代正式聚类和稳定性分析。

## 9. 诊断与稳健性

1. 画 Shepard 图：原不相似度对嵌入距离。
2. 比较 1、2、3 维 stress。
3. 检查经典 MDS 特征值和负值比例。
4. 改变距离度量并做 Procrustes/相关比较。
5. bootstrap 样本后对齐嵌入，评估结构稳定性。

## 10. 推荐可视化

- MDS1-MDS2 散点图。
- Shepard 图及拟合线。
- stress 随维度变化曲线。
- 经典 MDS 特征值谱。

## 11. 优势、局限与常见坑

**优势：** 可直接从距离矩阵出发，几何含义清晰，适合多种非原始特征距离。

**局限：** 二次存储，轴难解释，结果强依赖距离，低维可能严重失真。

**常见坑：** 不报告 stress；把方向当医学轴；将相似度直接当距离；忽略负特征值；凭二维视觉宣称分群。

## 12. 与相近方法的区别

- [[主成分分析（Principal Component Analysis, PCA）]]：对欧氏数据，经典 MDS 与 PCA 得分紧密相关；MDS 更自然地接收距离矩阵。
- [[Isomap（Isometric Mapping）]]：先估计近邻图测地距离，再调用经典 MDS。
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]：更强调局部概率邻域，不追求全局距离数值。
- 选择经验：研究对象由专门距离定义时考虑 MDS；原始连续特征线性压缩先用 PCA。

## 13. 医学研究中的典型应用

- 微生物组 beta 多样性排序图。
- 患者、症状或临床文本不相似度可视化。
- 遗传距离、生态距离与医院间模式比较。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| dissimilarity | 两对象差异程度，不一定是欧氏距离 |
| double centering | 从平方距离恢复中心化内积矩阵 |
| Gram matrix | 对象两两内积组成的矩阵 |
| stress | 原不相似度与嵌入距离的失配程度 |
| Shepard diagram | 不相似度与嵌入距离的散点图 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Isomap（Isometric Mapping）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 16. 参考资料

- Torgerson WS. Multidimensional scaling: I. Theory and method. *Psychometrika*. 1952;17:401-419.
- Kruskal JB. Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika*. 1964;29:1-27.
- scikit-learn Developers. `MDS` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) （访问日期：2026-07-09）

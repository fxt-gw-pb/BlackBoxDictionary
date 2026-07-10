---
title: 均值漂移聚类
english_name: Mean Shift Clustering
slug: mean-shift-clustering
aliases: [Mean Shift, mean shift clustering, 均值漂移, "均值漂移聚类（Mean Shift Clustering）"]
category: 聚类与无监督学习
subcategory: 密度聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 密度估计]
status: 已建
difficulty: intermediate
question_type: 密度峰发现与自动簇数聚类
data_type: [表格数据, 空间数据, 图像特征]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [meanShiftR]
---

# 均值漂移聚类（Mean Shift Clustering）

## 1. 方法概览

### 1.1 一句话本质

Mean Shift 让候选中心反复移动到局部邻域的加权均值，沿密度上升方向收敛到密度峰，并把到达同一峰的样本归为一簇。

### 1.2 定义

均值漂移聚类是一种非参数密度峰聚类方法。它以核密度估计为基础，用带宽规定观察尺度，不要求事先给定簇数；最终簇数由该尺度下保留下来的密度峰数量决定。

### 1.3 它主要解决什么问题

- 簇数未知，但局部高密度模式有意义。
- 希望找到密度峰而非均值质心。
- 图像、空间或低维连续数据存在多个模式。

### 1.4 直觉与类比

每个起点都站在密度地形上，反复走向周围人群的平均位置。若多个起点最后爬上同一座山峰，它们属于同一簇。

## 2. 核心思想与原理

### 2.1 均值偏移就是上坡方向

核密度的梯度可写成与“局部加权均值减当前位置”同方向的量。移动到局部均值，等价于向密度更高处迭代。

### 2.2 带宽定义分群尺度

小带宽保留细小峰，簇数多且对噪声敏感；大带宽把邻近峰平滑合并，簇数少。所谓“自动簇数”并非无参数，而是把 $K$ 的选择换成带宽选择。

### 2.3 峰合并与标签

不同种子可能收敛到非常接近的峰，算法需要合并近重复中心，再把样本分给相应峰。实现细节会影响边界点和孤立点标签。

## 3. 数学形式

### 3.1 核密度估计

$$
\hat f_h(x)=
\frac{1}{nh^p}
\sum_{i=1}^{n}
K\left(\frac{x-x_i}{h}\right)
$$

### 3.2 均值漂移向量

$$
m_h(x)=
\frac{\sum_i x_i g\left(\left\|\frac{x-x_i}{h}\right\|^2\right)}
{\sum_i g\left(\left\|\frac{x-x_i}{h}\right\|^2\right)}
-x
$$

### 3.3 更新

$$
x^{(t+1)}=x^{(t)}+m_h[x^{(t)}]
$$

平坦核实现中，更新就是半径 $h$ 邻域内样本的普通均值。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 低维密度可估 | 高维密度峰不稳定 | 降维和距离诊断 |
| 单一带宽有意义 | 不同尺度簇被合并或碎裂 | 带宽路径 |
| 特征尺度合理 | 大量纲变量主导邻域 | 标准化敏感性 |
| 样本覆盖足够 | 假峰或漏峰 | bootstrap 稳定性 |

## 4. 手把手算例

一维样本：

$$
x=(0,1,2,8,9)
$$

使用半径 $h=2$ 的平坦核。

**Step 1：从种子 0 出发。** 距离 0 不超过 2 的点为 $\{0,1,2\}$：

$$
x^{(1)}=\frac{0+1+2}{3}=1
$$

在 1 处的邻域仍为 $\{0,1,2\}$，均值仍是 1，因此收敛到峰 1。

**Step 2：从种子 2 出发。** 它的邻域同样是 $\{0,1,2\}$，一步移动到 1，也归入第一峰。

**Step 3：从种子 8 出发。** 邻域为 $\{8,9\}$：

$$
x^{(1)}=\frac{8+9}{2}=8.5
$$

再次计算仍为 8.5，形成第二峰。

**结论：** 五个起点最终汇入 1 和 8.5 两个密度峰，因此得到两个簇。若把带宽增大到足以连接两组，两个峰可能被平滑成一个。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续特征、空间坐标或图像位置/颜色特征。
- 多量纲特征需标准化；高维通常先做有依据的表征压缩。
- 缺失需预处理，普通方法不建模删失或重复测量。

### 5.2 输入与产出

输入为带宽、种子、核和收敛标准。输出为密度峰中心、簇数和样本标签；标准实现不提供概率隶属度。

## 6. 适用场景

- 低维、簇数未知、密度峰具有实际含义。
- 图像分割、空间热点和模式发现。
- 不适合超大高维数据、不同簇尺度悬殊或需要概率归属的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
bandwidth = estimate_bandwidth(
    X_s,
    quantile=0.2,
    n_samples=min(2000, len(X_s)),
)

model = MeanShift(
    bandwidth=bandwidth,
    bin_seeding=True,
    min_bin_freq=5,
    cluster_all=False,
    n_jobs=-1,
)
cluster = model.fit_predict(X_s)
centers = model.cluster_centers_
print(len(centers), model.n_iter_)
```

### 7.2 R

```r
library(meanShiftR)

x <- scale(df[, c("X", "Y", "Intensity", "Texture", "LesionScore")])
fit <- meanShift(
  as.matrix(x),
  bandwidth = 1.2
)

cluster <- fit$assignment
table(cluster)
```

不同 R 包的函数名与输出结构差异较大，使用前应以所安装版本的帮助文档为准。

## 8. 结果如何解释

- 中心是密度模式，不一定等于簇内算术均值。
- 簇数是给定带宽下的结果，不是唯一真实簇数。
- `cluster_all=False` 时，无法归入峰邻域的点可标为 -1。
- 临床解释应表述为“在该尺度下的高密度模式”。

## 9. 诊断与稳健性

1. 画带宽-簇数和带宽-中心位置路径。
2. bootstrap 后比较峰数量与位置。
3. 检查孤立点和小簇是否由噪声造成。
4. 比较标准化、变量集和降维方案。
5. 与 DBSCAN、K-means 和 KDE 结果对照。

## 10. 推荐可视化

- 核密度曲线或热图与密度峰。
- 样本移动轨迹和最终中心。
- 带宽变化下簇数阶梯图。
- 图像或地图上的聚类区域。

## 11. 优势、局限与常见坑

**优势：** 不预设簇数，直接寻找密度峰，可用于任意峰形的吸引盆。

**局限：** 带宽敏感，高维和大样本昂贵，不同尺度簇难兼顾。

**常见坑：** 把自动簇数当无参数；直接接受默认带宽；未标准化；把每个小峰都命名为医学亚型；忽略带宽估计本身的计算成本。

## 12. 与相近方法的区别

- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]：按密度连通定义簇，Mean Shift 按密度峰吸引盆定义簇。
- [[K-means聚类（K-means Clustering）]]：固定 $K$ 并优化平方距离。
- [[核密度估计（Kernel Density Estimation, KDE）]]：KDE 估密度，Mean Shift 使用密度梯度找峰并分群。
- 选择经验：低维模式发现关注“峰”时选 Mean Shift，关注噪声与连通形状时选 DBSCAN。

## 13. 医学研究中的典型应用

- 医学图像像素或区域的空间-颜色联合分割。
- 病例地理分布与空间组学密度峰探索。
- 连续表型中的常见模式中心识别。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| mode | 概率密度的局部最大点 |
| bandwidth | 决定局部邻域和平滑尺度的参数 |
| mean shift vector | 局部加权均值减当前位置 |
| basin of attraction | 会收敛到同一密度峰的起点集合 |
| bin seeding | 用网格箱中心减少初始种子数量 |

## 15. 相关方法

- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]
- [[K-means聚类（K-means Clustering）]]
- [[核密度估计（Kernel Density Estimation, KDE）]]
- [[谱聚类（Spectral Clustering）]]

## 16. 参考资料

- Fukunaga K, Hostetler L. The estimation of the gradient of a density function, with applications in pattern recognition. *IEEE Trans Inf Theory*. 1975;21(1):32-40.
- Comaniciu D, Meer P. Mean shift: a robust approach toward feature space analysis. *IEEE Trans Pattern Anal Mach Intell*. 2002;24(5):603-619.
- scikit-learn Developers. `MeanShift` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) （访问日期：2026-07-09）

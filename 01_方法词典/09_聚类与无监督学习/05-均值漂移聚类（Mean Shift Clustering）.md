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

### 1.1 定义

均值漂移聚类是一种基于密度峰的无监督方法。它通过不断把样本点移动到邻域内样本的加权均值位置，使点沿着密度上升方向收敛到局部密度峰，最终把收敛到同一峰的点归为一簇。

### 1.2 它主要解决什么问题

- 研究问题：不预设簇数时，数据中有哪些高密度中心或模式。
- 适用任务：密度峰聚类、图像分割、空间热点探索、模式发现。
- 常见医学场景：影像像素/区域分割，空间流行病学热点探索，连续临床表型中的高密度患者群识别。

### 1.3 直觉理解

均值漂移像让每个样本沿着“人群更密集”的方向爬坡。多个样本如果最后爬到同一个密度山峰，就被视为属于同一簇。

## 2. 数学形式

### 2.1 核心公式

给定核函数 $K$ 和带宽 $h$，核密度估计为：

$$
\hat f(x)=\frac{1}{nh^p}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right)
$$

均值漂移向量可写为：

$$
m(x)=
\frac{\sum_{i=1}^{n}x_i g\left(\left\|\frac{x-x_i}{h}\right\|^2\right)}
{\sum_{i=1}^{n}g\left(\left\|\frac{x-x_i}{h}\right\|^2\right)}
-x
$$

迭代更新：

$$
x^{(t+1)}=x^{(t)}+m(x^{(t)})
$$

### 2.2 参数或统计量含义

- `bandwidth`：核窗口带宽，控制邻域尺度。
- 核函数：用于给邻域内样本赋权。
- mode / density peak：样本最终收敛到的密度峰。
- cluster center：聚类中心，即密度峰位置。
- `bin_seeding`：是否用离散网格加速初始化。

### 2.3 关键假设

- 簇可被密度峰合理刻画。
- 一个合适的带宽能表达目标尺度。
- 特征空间中的距离和密度对研究问题有意义。
- 样本量足以估计局部密度。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续数值特征、空间坐标、图像颜色/位置特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：高维下密度估计困难，通常需要降维或选取少数关键特征。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先定义样本单位或特征表示。

### 3.2 示例表格

以影像区域特征为例：

| X | Y | Intensity | Texture | LesionScore |
| --- | --- | --- | --- | --- |
| 12 | 35 | 0.82 | 1.20 | 0.77 |
| 13 | 36 | 0.85 | 1.15 | 0.80 |
| 80 | 22 | 0.31 | 0.40 | 0.12 |
| 81 | 23 | 0.28 | 0.38 | 0.10 |

### 3.3 输入与产出

#### 输入

- 输入数据：连续型特征矩阵。
- 关键变量：带宽、核函数、停止阈值。
- 需要预处理的内容：缺失处理、标准化、带宽估计、异常值检查。

#### 产出

- 模型对象/统计结果：簇标签、密度峰中心、估计簇数。
- 参数估计：密度峰位置。
- 预测结果：可把新样本分配到最近密度峰，但标准实现主要用于拟合数据。
- 不确定性指标：带宽敏感性、簇中心稳定性、重采样一致性。

## 4. 适用场景

- 适合：簇数未知、希望寻找密度峰、数据维度较低或已降维的场景。
- 不适合：样本量很大且维度高、各簇尺度差异很大、需要明确概率归属的场景。
- 使用前需要特别检查的点：带宽选择、密度峰是否稳定、是否存在由于尺度差异造成的假峰。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("image_region_features.csv")
X = df[["X", "Y", "Intensity", "Texture", "LesionScore"]]

X_scaled = StandardScaler().fit_transform(X)
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=len(X_scaled))

fit = MeanShift(bandwidth=bandwidth, bin_seeding=True)
cluster = fit.fit_predict(X_scaled)

print("Clusters:", len(set(cluster)))
print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `meanShiftR`

```r
library(meanShiftR)

x <- scale(df[, c("X", "Y", "Intensity", "Texture", "LesionScore")])
fit <- meanShift(as.matrix(x), bandwidth = 1.2)

cluster <- fit$assignment
table(cluster)
```

## 6. 结果如何解释

- 核心结果看什么：密度峰位置、簇数量、各簇样本量、不同带宽下结果是否稳定。
- 每个主要参数如何解释：带宽越大，密度估计越平滑，簇数通常越少。
- 临床或医学意义如何表达：适合描述为“在所选尺度下发现若干高密度模式”，而不是直接宣称存在固定疾病亚型。
- 常见误读：自动得到簇数不代表该簇数就是唯一正确答案。

## 7. 推荐可视化

- 二维特征空间中的密度峰和聚类结果。
- 带宽变化下簇数曲线。
- 核密度热图。
- 图像或空间数据中的聚类区域叠加图。

## 8. 优势、局限与常见坑

### 优势

- 不需要预先指定簇数。
- 可发现密度峰和非参数模式。
- 对图像分割和空间热点直观。

### 局限

- 带宽选择非常关键。
- 高维密度估计困难。
- 大样本计算成本较高。

### 常见坑

- 不标准化多尺度特征。
- 把带宽估计默认值当成唯一合理选择。
- 在高维稀疏数据上直接使用而不先降维。

## 9. 与相近方法的区别

- 和 [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]] 的区别：DBSCAN 以密度连通定义簇，均值漂移以密度峰吸引盆定义簇。
- 和 [[K-means聚类（K-means Clustering）]] 的区别：K-means 需要预设 $K$，均值漂移通过密度峰自动形成簇。
- 和 [[核密度估计（Kernel Density Estimation, KDE）]] 的区别：KDE 是密度估计工具，均值漂移利用 KDE 的梯度思想做聚类。

## 10. 医学研究中的典型应用

- 医学影像分割中的颜色、纹理和空间特征聚类。
- 空间病例分布的高密度区域识别。
- 连续临床表型数据中的密度峰亚群探索。

## 11. 相关方法

- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]
- [[K-means聚类（K-means Clustering）]]
- [[核密度估计（Kernel Density Estimation, KDE）]]
- [[谱聚类（Spectral Clustering）]]

## 12. 参考资料

- Fukunaga K, Hostetler L. The estimation of the gradient of a density function, with applications in pattern recognition. *IEEE Transactions on Information Theory*. 1975;21(1):32-40.
- Comaniciu D, Meer P. Mean shift: a robust approach toward feature space analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 2002;24(5):603-619.
- scikit-learn Developers. `sklearn.cluster.MeanShift`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) （访问日期：2026-07-02）

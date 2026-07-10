---
title: 模糊C均值聚类
english_name: Fuzzy C-Means Clustering, FCM
slug: fuzzy-c-means-clustering-fcm
aliases: [FCM, fuzzy c-means, fuzzy c-means clustering, 模糊C均值, "模糊C均值聚类（Fuzzy C-Means Clustering, FCM）"]
category: 聚类与无监督学习
subcategory: 软聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 软聚类]
status: 已建
difficulty: intermediate
question_type: 模糊隶属度分群与边界样本识别
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督分群, 隶属度]
python_packages: [scikit-fuzzy]
r_packages: [e1071, ppclust]
---

# 模糊C均值聚类（Fuzzy C-Means Clustering, FCM）

## 1. 方法概览

### 1.1 一句话本质

FCM 用到各中心的相对距离为每个样本分配一组和为 1 的模糊隶属度，再用隶属度幂加权更新中心。

### 1.2 定义

模糊 C 均值是软原型聚类方法。它将 K-means 的硬标签替换为隶属度矩阵 $U$，允许患者同时部分属于多个簇；模糊系数 $m$ 控制归属的清晰程度。

### 1.3 它主要解决什么问题

- 患者处于连续谱或亚型过渡区。
- 希望识别归属不明确的边界病例。
- 硬聚类会丢失“更像哪个簇、像到什么程度”的信息。

### 1.4 直觉与类比

K-means 要求每个人只投一票；FCM 允许把一票拆开，例如对炎症型投 0.7、代谢型投 0.3。中心再根据这些分票的幂次权重更新。

## 2. 核心思想与原理

### 2.1 相对距离决定隶属度

一个点是否属于簇 1，不只看它离中心 1 多远，还看它相对其他中心有多远。位于两个中心正中间时，隶属度自然相等。

### 2.2 模糊系数

$m$ 接近 1 时隶属度趋于硬分配；$m$ 增大时分配更平均。过大的 $m$ 会让所有病例都显得模糊，削弱簇结构。

### 2.3 隶属度不是概率

FCM 没有为数据指定生成概率分布，其隶属度由距离目标函数定义。它不能像 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的责任概率那样解释为模型后验概率。

## 3. 数学形式

### 3.1 约束与目标

$$
\sum_{k=1}^{C}u_{ik}=1,\qquad
0\le u_{ik}\le1
$$

$$
J_m=
\sum_{i=1}^{n}\sum_{k=1}^{C}
u_{ik}^{m}\|x_i-c_k\|^2,\qquad m\gt1
$$

### 3.2 中心更新

$$
c_k=
\frac{\sum_i u_{ik}^{m}x_i}
{\sum_i u_{ik}^{m}}
$$

### 3.3 隶属度更新

$$
u_{ik}=
\frac{1}
{\sum_{j=1}^{C}
\left(
\frac{\|x_i-c_k\|}
{\|x_i-c_j\|}
\right)^{2/(m-1)}}
$$

若点恰好等于某中心，需要按算法约定处理零距离。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 中心型欧氏结构合理 | 非凸簇仍会分错 | 与密度/图方法比较 |
| $C$ 与 $m$ 合理 | 过分硬化或全部模糊 | validity 指标与路径 |
| 特征尺度可比 | 大量纲变量支配隶属度 | 标准化敏感性 |
| 离群点有限 | 中心与隶属度被拉偏 | 稳健处理与病例审查 |

## 4. 手把手算例

一维有三个患者 $x=(2,5,8)$，初始中心 $c_1=0,c_2=10$，取 $m=2$。

对 $x=2$，到两个中心的距离为 2 和 8：

$$
u_{1}=
\frac{1}{1+(2/8)^2}
=\frac{16}{17}
\approx0.941
$$

$$
u_{2}=1-u_1\approx0.059
$$

对 $x=5$，两个距离相等，因此隶属度为 $(0.5,0.5)$。对 $x=8$，结果与 $x=2$ 对称，为 $(0.059,0.941)$。

**更新中心。** 因为 $m=2$，使用隶属度平方加权：

$$
c_1^{new}
=
\frac{
2(0.941^2)+5(0.5^2)+8(0.059^2)
}{
0.941^2+0.5^2+0.059^2
}
\approx2.676
$$

由对称性：

$$
c_2^{new}\approx7.324
$$

**结论：** 中间患者对两簇各贡献一部分；两端患者主要拉动靠近自己的中心，但并非完全不影响另一中心。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续数值特征，通常需标准化。
- 缺失需预处理；高维时距离可能趋同。
- 普通 FCM 不直接处理类别特征、删失或患者内相关性。

### 5.2 输入与产出

输入为簇数 $C$、模糊系数 $m$、初始隶属度和停止阈值。输出为中心、隶属度矩阵、硬标签、目标轨迹与模糊分区指标。

## 6. 适用场景

- 亚群边界模糊、连续表型或软图像分割。
- 需要重点识别混合表型和过渡病例。
- 不适合离群点很多、非凸簇明显或必须提供概率后验的任务。

## 7. 实现

### 7.1 Python

```python
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X).T
centers, membership, _, distances, objective, iterations, fpc = (
    fuzz.cluster.cmeans(
        X_s,
        c=3,
        m=2.0,
        error=0.005,
        maxiter=1000,
        seed=42,
    )
)

cluster = np.argmax(membership, axis=0)
max_membership = membership.max(axis=0)
entropy = -(membership * np.log(membership + 1e-12)).sum(axis=0)
print(fpc, iterations, max_membership[:5], entropy[:5])
```

### 7.2 R

```r
library(e1071)

x <- scale(df[, c("CRP", "IL6", "BMI", "TG", "HDL")])
fit <- cmeans(
  x,
  centers = 3,
  m = 2,
  iter.max = 1000,
  dist = "euclidean",
  method = "cmeans"
)

cluster <- fit$cluster
membership <- fit$membership
max_membership <- apply(membership, 1, max)
head(membership)
```

## 8. 结果如何解释

- 隶属度描述距离目标下的相对归属，不是发生概率。
- 最大隶属度低或熵高的病例处于边界或整体远离中心。
- 中心是模糊权重均值，不能直接视作真实疾病原型。
- 若最终只保留硬标签，会丢失 FCM 最有价值的信息。

## 9. 诊断与稳健性

1. 比较不同 $C$ 与 $m$ 的 FPC、partition entropy 和稳定性。
2. 多初始化并比较目标函数与中心。
3. 画最大隶属度和熵分布。
4. 改变标准化、变量集和异常值处理。
5. 与 K-means 和 GMM 的标签及软归属比较。

## 10. 推荐可视化

- 隶属度热图。
- 最大隶属度或隶属度熵分布。
- 二维投影按主簇着色、按确定性调透明度。
- 模糊中心特征热图。

## 11. 优势、局限与常见坑

**优势：** 保留边界信息，表达过渡状态，输出比硬聚类丰富。

**局限：** 需给定 $C$ 和 $m$，对尺度、初始化和离群点敏感，仍偏好中心型簇。

**常见坑：** 把隶属度叫后验概率；只汇报硬标签；默认 $m=2$ 不做敏感性；用全部数据挑参数后夸大稳定性。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：FCM 将硬指示变量放松为连续隶属度。
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]：GMM 责任度是生成概率模型后验，FCM 隶属度是距离优化权重。
- [[期望最大化算法（Expectation-Maximization Algorithm, EM）]]：EM 是概率模型估计算法，FCM 是具体模糊目标算法。
- 选择经验：需要概率密度与协方差选 GMM；只需距离型软边界可选 FCM。

## 13. 医学研究中的典型应用

- 炎症、代谢或免疫表型连续谱。
- 疾病亚型间过渡患者识别。
- 医学影像组织或病灶软分割。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| membership | 样本对某簇的模糊隶属度 |
| fuzzifier | 控制隶属度软硬程度的参数 $m$ |
| fuzzy centroid | 由隶属度幂加权得到的中心 |
| partition coefficient | 衡量分区清晰程度的指标 |
| partition entropy | 衡量隶属度不确定性的熵指标 |

## 15. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[期望最大化算法（Expectation-Maximization Algorithm, EM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 16. 参考资料

- Dunn JC. A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. *J Cybern*. 1973;3(3):32-57.
- Bezdek JC. *Pattern Recognition with Fuzzy Objective Function Algorithms*. Springer; 1981.
- scikit-fuzzy Developers. Fuzzy C-means documentation. [https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_cmeans.html](https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_cmeans.html) （访问日期：2026-07-09）

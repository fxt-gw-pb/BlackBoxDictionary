---
title: Mini-Batch K-means聚类
english_name: Mini-Batch K-means Clustering
slug: mini-batch-k-means-clustering
aliases: [Mini-Batch K-means, mini-batch k-means clustering, 小批量K均值聚类, "Mini-Batch K-means聚类（Mini-Batch K-means Clustering）"]
category: 聚类与无监督学习
subcategory: 原型聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 大规模学习]
status: 已建
difficulty: basic
question_type: 大规模硬聚类与快速质心学习
data_type: [表格数据, 高维特征矩阵, 图像像素矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [ClusterR]
---

# Mini-Batch K-means聚类（Mini-Batch K-means Clustering）

## 1. 方法概览

### 1.1 一句话本质

Mini-Batch K-means 每次只抽一小批样本近似更新质心，以少量精度和随机波动换取速度、低内存与增量处理能力。

### 1.2 定义

Mini-Batch K-means 是 [[K-means聚类（K-means Clustering）]] 的随机近似版本。它保留最近质心分配和均值原型，但不在每轮扫描全部数据，而用小批量递推质心。

### 1.3 它主要解决什么问题

- 完整 K-means 每轮扫描海量数据成本过高。
- 数据无法一次装入内存或持续到达。
- 需要快速原型、压缩或初步分群。

### 1.4 直觉与类比

标准 K-means 每次重算全班平均；Mini-Batch 每次只看一小组新样本，按累计人数微调平均。看得足够多后，质心通常接近完整数据结果。

## 2. 核心思想与原理

### 2.1 随机近似

小批量若近似代表总体，其更新方向平均而言接近完整 K-means。单次更新噪声更大，但计算远少于全量扫描。

### 2.2 递推均值

对某簇已累计 $n_k$ 个样本，收到新样本 $x$ 后：

$$
\mu_k^{new}=
\mu_k+\frac{1}{n_k+1}(x-\mu_k)
$$

无需保存该簇所有历史样本。

### 2.3 继承 K-means 偏好

它改变的是优化方式，不是聚类目标。因此仍偏好球形、相近尺度簇，仍受标准化、$K$ 和离群点影响。

## 3. 数学形式

### 3.1 目标函数

目标仍近似最小化：

$$
\sum_{i=1}^{n}\|x_i-\mu_{c_i}\|_2^2
$$

### 3.2 小批量分配

对 $x_i\in B_t$：

$$
c_i=
\operatorname*{arg\,min}_{k}
\|x_i-\mu_k^{(t)}\|_2^2
$$

### 3.3 随机更新

$$
\mu_k^{(t+1)}
=(1-\eta_{kt})\mu_k^{(t)}
+\eta_{kt}\bar x_{k,B_t}
$$

$\eta_{kt}$ 通常随该中心累计更新次数下降。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 批次近似随机代表总体 | 质心随批次顺序漂移 | shuffle 与多随机种子 |
| batch 足够覆盖各簇 | 小簇中心被忽略 | 批量大小敏感性 |
| K-means 几何合理 | 近似得再快也切错结构 | 与替代算法对比 |
| 预处理一致 | 流式批次尺度变化 | 固定 scaler 与漂移监控 |

## 4. 手把手算例

一维数据分两批到达，初始中心为 $\mu_1=0,\mu_2=10$。

第一批：

$$
B_1=\{1,9\}
$$

1 分给中心 1，9 分给中心 2。每个中心第一次接收样本，更新为：

$$
\mu_1=1,\qquad\mu_2=9
$$

第二批：

$$
B_2=\{2,8\}
$$

2 分给第一簇。第一簇累计计数从 1 变 2：

$$
\mu_1^{new}
=1+\frac12(2-1)
=1.5
$$

同理：

$$
\mu_2^{new}
=9+\frac12(8-9)
=8.5
$$

**结论：** 只看两个小批次就得到完整四点 $\{1,2,8,9\}$ 的质心 1.5 和 8.5。一般数据中批次只提供近似，因此需检查随机波动。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 大规模连续数值矩阵、稀疏向量或像素特征。
- 缺失需预处理，尺度转换必须跨批次保持一致。
- 普通方法不直接建模患者内重复测量。

### 5.2 输入与产出

输入为 $K$、batch size、初始化、停止和低计数中心重分配规则。输出为近似质心、硬标签和 inertia。

## 6. 适用场景

- 大型 EHR、图像 patch、文本向量或流式数据。
- 可接受略低精度，优先考虑速度与内存。
- 小样本或非球形簇没有必要使用它。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

model = MiniBatchKMeans(
    n_clusters=4,
    batch_size=1024,
    init="k-means++",
    n_init=20,
    max_no_improvement=20,
    reassignment_ratio=0.01,
    random_state=42,
)
cluster = model.fit_predict(X_s)
centers = scaler.inverse_transform(model.cluster_centers_)
print(model.inertia_, model.n_steps_)
```

### 7.2 R

```r
library(ClusterR)

x <- scale(df[, c(
  "Age", "BMI", "VisitCount", "MedicationCount", "LabAbnormalCount"
)])

fit <- MiniBatchKmeans(
  x,
  clusters = 4,
  batch_size = 512,
  num_init = 20,
  early_stop_iter = 20
)

cluster <- predict_MBatchKMeans(x, fit$centroids)
table(cluster)
```

## 8. 结果如何解释

- 中心仍是原型，但受抽样顺序和近似更新影响。
- `batch_size` 越大通常越接近全量更新，但速度优势变小。
- inertia 应与相同预处理下的标准 K-means 比较。
- 快速分群不增加医学真实性，簇仍需画像与外部复现。

## 9. 诊断与稳健性

1. 比较不同 batch size 的运行时间、inertia 和标签一致性。
2. 使用多个随机种子评估质心波动。
3. 在可承受子样本上与完整 K-means 对照。
4. 检查小簇是否被重分配或消失。
5. 流式环境监控特征分布与中心漂移。

## 10. 推荐可视化

- batch size-运行时间-inertia 曲线。
- Mini-Batch 与完整 K-means 的共同聚类矩阵。
- 质心随批次更新轨迹。
- 标准化簇中心热图。

## 11. 优势、局限与常见坑

**优势：** 快、内存低、可增量更新，适合大规模质心学习。

**局限：** 有随机近似误差，小簇可能更新不足，继承 K-means 全部几何限制。

**常见坑：** batch 太小；数据未打乱；预处理跨批次变化；不与全量子样本比较；为大数据速度牺牲后仍过度解释亚型。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：全量更新通常更精确，Mini-Batch 更可扩展。
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]：BIRCH 建 CF 树微簇，Mini-Batch 直接随机更新质心。
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]：GMM 是概率软聚类。
- 选择经验：数据可装入内存时先用 K-means；大规模或增量处理时再用 Mini-Batch。

## 13. 医学研究中的典型应用

- 百万级患者 EHR 表型初筛。
- 医学图像像素、patch 或嵌入向量压缩。
- 可穿戴设备与长期监测片段的在线原型学习。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| mini-batch | 每次参与更新的一小批样本 |
| stochastic update | 基于随机样本近似全量更新 |
| running mean | 不保存历史数据的递推均值 |
| reassignment | 将长期低计数中心重新放置 |
| `partial_fit` | 用新批次增量更新模型的接口 |

## 15. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 16. 参考资料

- Sculley D. Web-scale k-means clustering. *WWW*. 2010:1177-1178.
- scikit-learn Developers. `MiniBatchKMeans` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) （访问日期：2026-07-09）
- CRAN. Package `ClusterR`. [https://cran.r-project.org/package=ClusterR](https://cran.r-project.org/package=ClusterR) （访问日期：2026-07-09）

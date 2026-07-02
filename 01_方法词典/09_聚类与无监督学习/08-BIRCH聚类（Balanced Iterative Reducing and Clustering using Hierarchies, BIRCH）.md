---
title: BIRCH聚类
english_name: Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH
slug: balanced-iterative-reducing-and-clustering-using-hierarchies-birch
aliases: [BIRCH, Birch clustering, "BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）"]
category: 聚类与无监督学习
subcategory: 大规模聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 大规模学习, 层次聚类]
status: 已建
difficulty: intermediate
question_type: 大规模数据摘要与层次聚类
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [stream]
---

# BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）

## 1. 方法概览

### 1.1 定义

BIRCH 是一种面向大规模数据的聚类算法。它通过簇特征树（CF tree）把大量样本压缩成可管理的微簇摘要，再对这些摘要进行聚类。

### 1.2 它主要解决什么问题

- 研究问题：样本量很大时，如何在有限内存下进行快速聚类。
- 适用任务：大规模表格数据聚类、流式或批量数据摘要、预聚类后再精细聚类。
- 常见医学场景：大规模 EHR 患者表型预聚类，长期监测数据压缩，海量影像 patch 特征预分组。

### 1.3 直觉理解

BIRCH 不把每个样本都长期保存在内存里，而是把相近样本概括成一个“微簇”。这些微簇记录样本数、线性和、平方和，足以计算中心和半径，再用树结构组织起来。

## 2. 数学形式

### 2.1 核心公式

一个簇特征（Cluster Feature, CF）定义为三元组：

$$
CF=(N, LS, SS)
$$

其中：

$$
LS=\sum_{i=1}^{N}x_i,\quad SS=\sum_{i=1}^{N}x_i^\top x_i
$$

簇中心为：

$$
\mu=\frac{LS}{N}
$$

簇半径可写为：

$$
R=\sqrt{\frac{SS}{N}-\left\|\frac{LS}{N}\right\|^2}
$$

两个 CF 可直接相加：

$$
CF_1+CF_2=(N_1+N_2,\ LS_1+LS_2,\ SS_1+SS_2)
$$

### 2.2 参数或统计量含义

- CF：微簇摘要，包含样本数、线性和、平方和。
- CF tree：层次化组织微簇的平衡树。
- `threshold`：叶节点微簇允许的最大半径或直径。
- `branching_factor`：每个非叶节点最多包含的子节点数。
- `n_clusters`：最终全局聚类的目标簇数；也可不指定，只输出子簇。

### 2.3 关键假设

- 数据可由大量局部紧凑微簇近似。
- 欧氏空间中的中心和半径能表达局部结构。
- 阈值设置能在压缩率和信息保留之间取得平衡。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型数值特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：大样本特征矩阵。
- 是否适合高维数据：可用，但高维距离和半径解释会变弱。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先构造样本级输入或窗口级特征。

### 3.2 示例表格

以大规模随访摘要特征为例：

| VisitCount | MeanSBP | MeanBMI | LabAbnormalCount | MedicationCount |
| --- | --- | --- | --- | --- |
| 18 | 146 | 31.2 | 6 | 8 |
| 4 | 118 | 23.6 | 1 | 2 |
| 12 | 139 | 28.8 | 4 | 5 |
| 3 | 112 | 22.4 | 0 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：大规模数值型特征矩阵。
- 关键变量：阈值、分支因子、最终簇数。
- 需要预处理的内容：缺失处理、标准化、异常值处理、变量选择。

#### 产出

- 模型对象/统计结果：微簇、CF 树、最终簇标签。
- 参数估计：微簇中心、半径和样本数。
- 预测结果：可把新样本分配到最近子簇或最终簇。
- 不确定性指标：阈值敏感性、簇稳定性、与 K-means 等方法的一致性。

## 4. 适用场景

- 适合：样本量很大、需要先压缩再聚类、数据可由局部紧凑微簇描述的场景。
- 不适合：簇形状非常复杂、离群点很多、特征主要为类别变量的场景。
- 使用前需要特别检查的点：threshold 是否导致过度合并或过度碎片化，微簇数量是否可解释。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import Birch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ehr_summary_features.csv")
X = df[["VisitCount", "MeanSBP", "MeanBMI", "LabAbnormalCount", "MedicationCount"]]

model = make_pipeline(
    StandardScaler(),
    Birch(threshold=0.6, branching_factor=50, n_clusters=4)
)
cluster = model.fit_predict(X)

print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `stream`

```r
library(stream)

x <- scale(df[, c("VisitCount", "MeanSBP", "MeanBMI", "LabAbnormalCount", "MedicationCount")])
stream <- DSD_Memory(x)
birch <- DSC_BIRCH(threshold = 0.6, branching = 50)
update(birch, stream, n = nrow(x))

micro <- get_centers(birch)
head(micro)
```

## 6. 结果如何解释

- 核心结果看什么：微簇数量、最终簇中心、各簇样本量、不同阈值下结果变化。
- 每个主要参数如何解释：`threshold` 越小，微簇越细；越大，压缩越强但可能合并不同结构。
- 临床或医学意义如何表达：BIRCH 更常作为大规模聚类的预处理或快速分群工具，最终簇仍需用临床特征画像解释。
- 常见误读：CF 树是计算摘要结构，不等同于医学上的层级病程结构。

## 7. 推荐可视化

- 阈值与微簇数量关系图。
- 最终簇在 PCA/UMAP 空间中的分布。
- 各簇特征中心热图。
- 微簇大小分布图。

## 8. 优势、局限与常见坑

### 优势

- 适合大规模数据和有限内存场景。
- 可增量构建局部数据摘要。
- 可与其他聚类算法组合使用。

### 局限

- 对阈值敏感。
- 对非球形、复杂形状簇不如密度或图方法灵活。
- 高维空间中半径和距离解释变弱。

### 常见坑

- 阈值不调参，直接接受默认微簇结构。
- 忽略离群点对 CF 树的影响。
- 把预聚类结果当作最终临床亚型。

## 9. 与相近方法的区别

- 和 [[层次聚类（Hierarchical Clustering）]] 的区别：BIRCH 用 CF 树做大规模数据压缩，传统层次聚类通常直接基于样本距离合并。
- 和 [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]] 的区别：Mini-Batch K-means 用小批量更新质心，BIRCH 先生成微簇摘要。
- 和 [[K-means聚类（K-means Clustering）]] 的区别：BIRCH 可作为 K-means 前的压缩步骤，也可直接输出聚类。

## 10. 医学研究中的典型应用

- 大规模 EHR 患者表型预聚类。
- 海量医学影像 patch 特征摘要。
- 可穿戴设备长期监测数据的模式压缩。

## 11. 相关方法

- [[层次聚类（Hierarchical Clustering）]]
- [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]]
- [[K-means聚类（K-means Clustering）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 12. 参考资料

- Zhang T, Ramakrishnan R, Livny M. BIRCH: an efficient data clustering method for very large databases. *SIGMOD*. 1996:103-114.
- scikit-learn Developers. `sklearn.cluster.Birch`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html) （访问日期：2026-07-02）
- Hahsler M, Piekenbrock M, Doran D. `stream`: An extensible framework for data stream clustering research with R. *Journal of Statistical Software*. 2019;76(14):1-50.

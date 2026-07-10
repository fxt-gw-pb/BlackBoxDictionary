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

### 1.1 一句话本质

BIRCH 把相近样本在线压缩为可相加的 CF 微簇，再在微簇中心上做全局聚类，从而避免反复保存和比较全部原始点。

### 1.2 定义

BIRCH 是面向大规模数值数据的增量聚类算法。它构建平衡 CF 树，每个叶微簇只保存样本数、线性和与平方和；阈值决定新点被吸收还是生成新微簇。

### 1.3 它主要解决什么问题

- 全样本距离矩阵无法存储。
- 需要单遍或分批压缩海量样本。
- 希望先形成局部微簇，再用层次聚类等方法得到最终簇。

### 1.4 直觉与类比

不保存每一张病历，而把相近患者装进摘要盒：盒上记录人数、指标总和和平方和。新患者沿树找到最近盒，若加入后盒仍够紧就吸收，否则新开一个盒。

## 2. 核心思想与原理

### 2.1 可加的充分摘要

CF 三元组足以恢复微簇中心和半径，两个微簇可直接相加。算法因此能增量更新而不访问所有历史点。

### 2.2 CF 树控制内存

每个节点只能容纳有限子簇，超出 branching factor 时节点分裂。threshold 控制叶微簇紧密程度，二者共同决定树大小。

### 2.3 两阶段聚类

第一阶段压缩为微簇；第二阶段可对微簇中心做层次聚类或其他全局聚类。CF 树是计算结构，不等于最终医学层级。

## 3. 数学形式

### 3.1 CF 三元组

对 $N$ 个 $p$ 维样本：

$$
CF=(N,LS,SS)
$$

$$
LS=\sum_{i=1}^{N}x_i,\qquad
SS=\sum_{i=1}^{N}x_i^\top x_i
$$

### 3.2 中心与半径

$$
\mu=\frac{LS}{N}
$$

$$
R=
\sqrt{\frac{SS}{N}-\left\|\frac{LS}{N}\right\|^2}
$$

### 3.3 可加性

$$
CF_1+CF_2=
(N_1+N_2,\ LS_1+LS_2,\ SS_1+SS_2)
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 局部结构可由球形微簇近似 | 非凸结构被压坏 | 与原数据子样本对照 |
| threshold 合理 | 过度合并或微簇爆炸 | 阈值-微簇数曲线 |
| 特征尺度一致 | 半径由大量纲变量决定 | 固定标准化 |
| 数据顺序影响可接受 | 在线树结构波动 | 多顺序重放 |

## 4. 手把手算例

一维微簇 A 含样本 $(1,2,3)$：

$$
N_A=3,\quad LS_A=6,\quad SS_A=1^2+2^2+3^2=14
$$

其中心和半径：

$$
\mu_A=6/3=2
$$

$$
R_A=\sqrt{14/3-2^2}
=\sqrt{2/3}\approx0.816
$$

微簇 B 含 $(8,9)$：

$$
CF_B=(2,17,145),\quad
\mu_B=8.5,\quad R_B=0.5
$$

**尝试合并。**

$$
CF_{A+B}=(5,23,159)
$$

$$
\mu_{A+B}=23/5=4.6
$$

$$
R_{A+B}
=\sqrt{159/5-4.6^2}
=\sqrt{10.64}
\approx3.262
$$

若 threshold 为 1，A 和 B 各自可作为紧凑微簇，但合并半径 3.262 超过阈值，因此必须分开保存。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 大规模连续数值特征或稀疏数值向量。
- 缺失需预处理，标准化规则必须在所有批次一致。
- 高维距离仍可能失效，BIRCH 不自动解决维度灾难。

### 5.2 输入与产出

输入为 threshold、branching factor、最终聚类器和数据批次。输出为 CF 树、微簇中心、微簇标签与最终样本标签。

## 6. 适用场景

- 大规模、流式或内存受限的数值数据。
- 原始点可由局部紧凑微簇近似。
- 不适合复杂非凸簇、类别特征为主或对极小罕见簇要求很高的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

model = Birch(
    threshold=0.6,
    branching_factor=50,
    n_clusters=4,
)

for batch in range(0, len(X_s), 2000):
    model.partial_fit(X_s[batch:batch + 2000])
model.partial_fit()  # 完成全局聚类步骤

cluster = model.predict(X_s)
micro_centers = model.subcluster_centers_
print(len(micro_centers))
```

### 7.2 R

```r
library(stream)

x <- scale(df[, c(
  "VisitCount", "MeanSBP", "MeanBMI",
  "LabAbnormalCount", "MedicationCount"
)])

source <- DSD_Memory(x)
model <- DSC_BIRCH(
  threshold = 0.6,
  branching = 50,
  maxLeaf = 100
)
update(model, source, n = nrow(x))

micro_centers <- get_centers(model)
head(micro_centers)
```

## 8. 结果如何解释

- 微簇是压缩单元，不一定是最终临床簇。
- threshold 越小，微簇越多、保真更高、内存更大。
- 最终 `n_clusters` 在微簇中心上运行，不等于 CF 树叶数。
- 数据顺序可影响树，需报告流式顺序和随机化策略。

## 9. 诊断与稳健性

1. 画 threshold 与微簇数、内存和下游指标的关系。
2. 随机打乱数据顺序并重复建树。
3. 在可承受子样本上与全量 K-means 或层次聚类比较。
4. 检查微簇半径、大小和极小簇。
5. 监控流式特征漂移与树增长。

## 10. 推荐可视化

- threshold-微簇数量曲线。
- 微簇大小和半径分布。
- 微簇中心及最终簇的二维投影。
- CF 树深度、节点数和内存随批次变化图。

## 11. 优势、局限与常见坑

**优势：** 内存高效、可增量、CF 可加，适合先压缩再聚类。

**局限：** threshold 和顺序敏感，偏好紧凑微簇，高维与非凸结构仍困难。

**常见坑：** 把 CF 树当临床层级；忽略 `maxLeaf` 等结构参数；预处理跨批次变化；直接把微簇当最终亚型。

## 12. 与相近方法的区别

- [[层次聚类（Hierarchical Clustering）]]：直接基于样本距离合并，BIRCH 先用 CF 树压缩。
- [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]]：直接随机更新 $K$ 个质心；BIRCH 维护许多微簇摘要。
- [[K-means聚类（K-means Clustering）]]：可作为 BIRCH 压缩后的全局聚类器。
- 选择经验：需要可加摘要和流式压缩时选 BIRCH，仅需快速固定 $K$ 质心时选 Mini-Batch。

## 13. 医学研究中的典型应用

- 海量 EHR 患者画像预聚类。
- 医学影像 patch 与可穿戴时间窗口特征压缩。
- 数据流中的在线模式摘要和后续精细聚类。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| CF | 保存样本数、线性和、平方和的簇摘要 |
| micro-cluster | CF 树叶部的局部紧凑摘要 |
| threshold | 新点并入微簇后允许的最大半径 |
| branching factor | 每个非叶节点最多的子簇数 |
| global clustering | 在微簇中心上执行的最终聚类 |

## 15. 相关方法

- [[层次聚类（Hierarchical Clustering）]]
- [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]]
- [[K-means聚类（K-means Clustering）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 16. 参考资料

- Zhang T, Ramakrishnan R, Livny M. BIRCH: an efficient data clustering method for very large databases. *SIGMOD*. 1996:103-114.
- scikit-learn Developers. `Birch` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html) （访问日期：2026-07-09）
- Hahsler M, Piekenbrock M, Doran D. `stream`: An extensible framework for data stream clustering research with R. *J Stat Softw*. 2019;76(14):1-50.

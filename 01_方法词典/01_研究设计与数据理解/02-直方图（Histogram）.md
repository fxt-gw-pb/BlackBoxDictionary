---
title: 直方图
english_name: Histogram
slug: histogram
aliases: [histogram, 频率直方图, 频数直方图, "直方图（Histogram）"]
category: 研究设计与数据理解
subcategory: 分布可视化
tags: [医学统计, 数据科学, 描述统计, 分布探索, 可视化]
status: 已建
difficulty: basic
question_type: 单变量分布描述
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [matplotlib, numpy]
r_packages: [graphics]
---

# 直方图（Histogram）

## 1. 方法概览

### 1.1 定义

直方图把连续变量的取值范围切成若干等宽区间（bin），统计每个区间内的观测数（或频率），用相邻柱形展示数据的分布形状。

### 1.2 它主要解决什么问题

- 研究问题：一个连续变量长什么样——中心在哪、有多分散、是否偏态、有几个峰、有无异常值。
- 适用任务：分布探索、建模前的分布假设检查、异常值与截断的初步识别。
- 常见医学场景：年龄、血压、生物标志物浓度的分布描述与数据质控。

### 1.3 直觉理解

直方图是把散落的数据“堆”进一排格子里，格子越高说明落在该区间的观测越多。它是认识一份数据的第一张图，几乎所有分布判断都从这里开始。

## 2. 数学形式

### 2.1 核心公式

给定 bin 宽 $h$、区间 $B_k=[x_0+kh,\,x_0+(k+1)h)$，频数与频率密度为：

$$
n_k=\sum_{i=1}^{n}\mathbf{1}\{x_i\in B_k\},\qquad
\hat{f}(x)=\frac{n_k}{n\,h},\ x\in B_k
$$

密度化后柱面积之和为 1，可与 [[核密度估计（Kernel Density Estimation, KDE）]] 叠加比较。

### 2.2 参数或统计量含义

- $h$：bin 宽（或 bin 数），决定平滑程度。
- $n_k$：第 $k$ 个区间的频数。
- $\hat{f}(x)$：频率密度估计，使不同 bin 宽下纵轴可比。
- 常见 bin 规则：Sturges、Scott、Freedman-Diaconis。

### 2.3 关键假设

- 变量为可分箱的数值量。
- 结果对 bin 宽和起点敏感，无“唯一正确”的直方图。
- 样本量太小时形状不稳定。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：单个连续变量（可按分组分面）。
- 因变量类型：不区分因变量。
- 数据结构：一维数值数组。
- 是否适合高维数据：仅单变量；高维需分面或改用其他图。
- 是否适合缺失较多数据：需先剔除或说明缺失。
- 是否适合删失数据：删失需专门处理，不宜直接画。
- 是否适合重复测量数据：混合层级会掩盖结构，宜分层。

### 3.2 示例表格

| bin 区间 | 频数 | 频率 |
| --- | --- | --- |
| [110,120) | 24 | 0.12 |
| [120,130) | 58 | 0.29 |
| [130,140) | 71 | 0.36 |
| [140,150) | 47 | 0.23 |

### 3.3 输入与产出

#### 输入

- 输入数据：一维数值向量。
- 关键变量：待描述的连续变量、bin 设置。
- 需要预处理的内容：处理缺失、决定 bin 宽/规则、是否密度化。

#### 产出

- 模型对象/统计结果：各 bin 频数/频率。
- 参数估计：无（描述性）。
- 预测结果：无。
- 不确定性指标：可叠加分组或密度曲线辅助判断。

## 4. 适用场景

- 适合：单变量分布的快速探索与质控。
- 不适合：小样本、需要精确密度、比较多组分布（可用密度曲线或箱线图）。
- 使用前需要特别检查的点：bin 宽是否掩盖或制造了峰、坐标是否被异常值拉伸。

## 5. 实现

### 5.1 Python

常用包：

- `matplotlib`
- `numpy`

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(130, 12, size=400)
plt.hist(x, bins="fd", density=True, color="#B0A895", edgecolor="white")
plt.xlabel("收缩压"); plt.ylabel("频率密度")
plt.show()
```

### 5.2 R

常用包：

- `graphics`

```r
x <- rnorm(400, 130, 12)
hist(x, breaks = "FD", freq = FALSE,
     xlab = "收缩压", main = "")
```

## 6. 结果如何解释

- 核心结果看什么：中心位置、离散程度、偏态、峰数、异常值。
- 每个主要参数如何解释：右偏（长尾在右）常见于浓度/时间类指标；双峰提示可能存在亚群。
- 临床或医学意义如何表达：分布形状影响后续检验/模型选择（如是否需要变换）。
- 常见误读：把 bin 宽造成的锯齿当真实结构；用单一直方图比较多组分布。

## 7. 推荐可视化

- 直方图叠加核密度曲线。
- 分组分面直方图。
- 与箱线图/小提琴图并列查看。

## 8. 优势、局限与常见坑

### 优势

- 直观、计算简单，是分布探索的第一图。
- 可密度化以叠加理论分布或 KDE。
- 易于分组分面。

### 局限

- 强烈依赖 bin 宽与起点。
- 小样本下形状不稳。
- 多组比较能力弱。

### 常见坑

- 默认 bin 数不当导致误判分布。
- 未密度化就跨不同 bin 宽比较纵轴。
- 用直方图代替正态性检验下结论。

## 9. 与相近方法的区别

- 和 [[核密度估计（Kernel Density Estimation, KDE）]] 的区别：KDE 用平滑核给出连续密度，避免 bin 边界的锯齿。
- 和 [[经验分布函数（Empirical Cumulative Distribution Function, ECDF）]] 的区别：ECDF 无需分箱、无信息损失地展示累积分布。
- 和箱线图的区别：箱线图概括分位数、便于多组比较，但看不出多峰。

## 10. 医学研究中的典型应用

- 基线连续变量（年龄、BMI、检验值）的分布描述。
- 数据清洗中识别异常值、截断与录入错误。
- 判断是否需要对数变换等预处理。

## 11. 相关方法

- [[核密度估计（Kernel Density Estimation, KDE）]]
- [[经验分布函数（Empirical Cumulative Distribution Function, ECDF）]]

## 12. 参考资料

- Freedman D, Diaconis P. On the histogram as a density estimator: L2 theory. *Z Wahrscheinlichkeitstheorie*. 1981;57(4):453-476.
- Scott DW. *Multivariate Density Estimation: Theory, Practice, and Visualization*. 2nd ed. Wiley; 2015.
- Wickham H, Grolemund G. *R for Data Science*. O'Reilly; 2017.

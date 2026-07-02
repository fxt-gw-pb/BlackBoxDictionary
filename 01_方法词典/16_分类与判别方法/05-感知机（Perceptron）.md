---
title: 感知机
english_name: Perceptron
slug: perceptron
aliases: [perceptron, Rosenblatt perceptron, "感知机（Perceptron）"]
category: 分类与判别方法
subcategory: 线性分类
tags: [医学统计, 数据科学, 神经网络, 分类]
status: 已建
difficulty: basic
question_type: 线性二分类
data_type: [表格数据, 向量数据]
outcome_type: [二分类]
python_packages: [scikit-learn]
r_packages: []
---

# 感知机（Perceptron）

## 1. 方法概览

### 1.1 定义

感知机是最早期、最简单的神经网络分类模型之一。它通过一个线性判别函数把样本分到两个类别，本质上是“能否用一个超平面把两类样本分开”的学习问题。

### 1.2 它主要解决什么问题

- 研究问题：当二分类问题近似线性可分时，如何快速学习一个简单判别边界。
- 适用任务：线性二分类、在线学习、作为更复杂神经网络的入门基线。
- 常见医学场景：基于少量结构化指标的阳性/阴性初筛、简单风险分层、线性可分信号模式识别。

### 1.3 直觉理解

可以把感知机理解成一条会不断“挪位置”的分界线。每次遇到分错的样本，它就朝着把这个样本分对的方向微调权重，直到大多数样本被正确分开。

## 2. 数学形式

### 2.1 核心公式

给定输入向量 $x \in \mathbb{R}^p$，感知机的判别函数为：

$$
f(x) = \operatorname{sign}(w^\top x + b)
$$

其中预测标签通常写为：

$$
\hat y =
\begin{cases}
1, & w^\top x + b \ge 0 \\
0, & w^\top x + b < 0
\end{cases}
$$

经典更新规则为：

$$
w \leftarrow w + \eta (y - \hat y)x,\qquad
b \leftarrow b + \eta (y - \hat y)
$$

### 2.2 参数或统计量含义

- $w$：各输入特征的权重。
- $b$：偏置，决定判别边界的平移。
- $\eta$：学习率，控制每次更新步长。
- 误分类次数：常用来判断训练是否收敛。

### 2.3 关键假设

- 两类样本至少近似线性可分。
- 特征已数值化，且最好做过中心化或标准化。
- 更适合小到中等规模、结构较简单的问题。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量、二元变量或已编码的离散变量。
- 因变量类型：二分类。
- 数据结构：二维特征矩阵。
- 是否适合高维数据：可以，但对噪声和尺度较敏感。
- 是否适合缺失较多数据：不适合，通常需先插补。
- 是否适合删失数据：不适合。
- 是否适合重复测量数据：基础感知机不直接处理相关结构。

### 3.2 示例表格

以简单感染初筛为例：

| Temperature | WBC | CRP | HeartRate | Infection |
| --- | --- | --- | --- | --- |
| 38.8 | 13.2 | 52 | 112 | 1 |
| 36.7 | 6.4 | 3 | 76 | 0 |
| 39.1 | 14.8 | 67 | 118 | 1 |
| 36.5 | 5.9 | 2 | 72 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：样本特征矩阵和二分类标签。
- 关键变量：学习率、迭代轮数、是否加正则化。
- 需要预处理的内容：标准化、异常值检查、类别编码。

#### 产出

- 模型对象/统计结果：线性判别器参数。
- 参数估计：权重和偏置。
- 预测结果：二分类标签，部分实现也可返回决策函数值。
- 不确定性指标：通常依赖外部验证集准确率、召回率、AUC 等，而非参数置信区间。

## 4. 适用场景

- 适合：线性边界明显、需要简单快速基线模型的场景。
- 不适合：非线性关系复杂、类别边界高度重叠、需要概率输出和稳定校准的场景。
- 使用前需要特别检查的点：线性可分程度、特征尺度、类别不平衡。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = df[["Temperature", "WBC", "CRP", "HeartRate"]]
y = df["Infection"]

model = make_pipeline(
    StandardScaler(),
    Perceptron(max_iter=1000, eta0=0.1, random_state=42)
)
model.fit(X, y)
pred = model.predict(X)
```

### 5.2 R

常用包：

- 无统一主流基础实现，实践中更常用 `keras` 或 `torch` 复现单层线性分类器

```r
# R 中若只是做二分类基线，通常更直接使用 glm() 或 keras / torch
# 来复现单层神经元，而不是单独维护经典感知机训练流程。
```

## 6. 结果如何解释

- 核心结果看什么：分类错误率、召回率、特异度，以及决策边界是否稳定。
- 每个主要参数如何解释：权重正负代表特征对分类方向的影响，但大小受量纲影响明显。
- 临床或医学意义如何表达：更适合作为简单筛查规则的技术基线，而不是最终解释模型。
- 常见误读：训练集完全分开不代表模型泛化好，也不代表真实机制就是线性的。

## 7. 推荐可视化

- 二维投影下的决策边界图。
- 误分类样本散点图。
- 类别分布与混淆矩阵。

## 8. 优势、局限与常见坑

### 优势

- 结构简单，易于理解。
- 训练速度快。
- 是理解神经网络与在线学习的良好起点。

### 局限

- 只能处理线性可分或近似线性可分问题。
- 不直接提供概率输出。
- 对噪声和异常值较敏感。

### 常见坑

- 没有标准化就直接训练。
- 在高度不平衡数据上只看准确率。
- 把感知机误当成可处理复杂非线性关系的模型。

## 9. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] 的区别：Logistic 回归输出概率并通过似然估计训练，感知机更强调误分类驱动的更新。
- 和 [[支持向量机（Support Vector Machine, SVM）]] 的区别：SVM 追求最大间隔，感知机只要求分开样本，不控制边界间隔。
- 和 [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]] 的区别：感知机是单层线性判别器，MLP 通过隐藏层学习非线性映射。

## 10. 医学研究中的典型应用

- 少量实验室指标的二分类初筛。
- 简单生理信号模式的线性判别。
- 作为更复杂模型前的可解释基线。

## 11. 相关方法

- [[Logistic回归（Logistic Regression）]]
- [[支持向量机（Support Vector Machine, SVM）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]

## 12. 参考资料

- Rosenblatt F. The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*. 1958;65(6):386-408.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- scikit-learn Developers. `sklearn.linear_model.Perceptron`. [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) （访问日期：2026-07-02）

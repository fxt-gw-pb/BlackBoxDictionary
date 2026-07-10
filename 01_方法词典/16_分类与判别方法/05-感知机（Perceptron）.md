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

### 1.1 一句话本质

感知机从一个线性边界出发，每遇到误分类样本就把边界朝纠正该样本的方向移动。

### 1.2 定义

感知机是最早期、最简单的神经网络分类模型之一。它通过一个线性判别函数把样本分到两个类别，本质上是“能否用一个超平面把两类样本分开”的学习问题。

### 1.3 它主要解决什么问题

- 研究问题：当二分类问题近似线性可分时，如何快速学习一个简单判别边界。
- 适用任务：线性二分类、在线学习、作为更复杂神经网络的入门基线。
- 常见医学场景：基于少量结构化指标的阳性/阴性初筛、简单风险分层、线性可分信号模式识别。

### 1.4 直觉理解

可以把感知机理解成一条会不断“挪位置”的分界线。每次遇到分错的样本，它就朝着把这个样本分对的方向微调权重，直到大多数样本被正确分开。

## 2. 核心思想与原理

### 2.1 根本困难

需要从带标签样本中找到一个超平面，使正类位于一侧、负类位于另一侧。边界参数很多，而“哪条线能分开”无法靠逐项尝试高效确定。

### 2.2 关键洞察

只在犯错时学习：若正类落到负侧，就沿其特征向量增加权重；若负类落到正侧，就反向移动。正确样本不更新，使算法成为简单的在线误差修正规则。

### 2.3 收敛边界

若训练集线性可分，感知机收敛定理保证有限次更新后找到某个分离超平面；若类别重叠或存在标签噪声，权重可能持续振荡，必须限制轮数、使用 pocket 策略或改用带损失与正则化的模型。

## 3. 数学形式

### 3.1 核心公式

给定输入向量 $x \in \mathbb{R}^p$，感知机的判别函数为：

$$
f(x) = \operatorname{sign}(w^\top x + b)
$$

其中预测标签通常写为：

$$
\hat y =
\begin{cases}
1, & w^\top x + b \ge 0 \\
0, & w^\top x + b \lt 0
\end{cases}
$$

经典更新规则为：

$$
w \leftarrow w + \eta (y - \hat y)x,\qquad
b \leftarrow b + \eta (y - \hat y)
$$

若标签编码为 $y\in\{-1,+1\}$，则在 $y(w^\top x+b)\le 0$ 时可写成更经典的更新：

$$
w\leftarrow w+\eta yx,\qquad b\leftarrow b+\eta y
$$

### 3.3 参数或统计量含义

- $w$：各输入特征的权重。
- $b$：偏置，决定判别边界的平移。
- $\eta$：学习率，控制每次更新步长。
- 误分类次数：常用来判断训练是否收敛。

### 3.4 关键假设

- 两类样本至少近似线性可分。
- 特征已数值化，且最好做过中心化或标准化。
- 更适合小到中等规模、结构较简单的问题。

## 4. 手把手算例

令标签为 $\{-1,+1\}$，学习率 $\eta=1$，初始 $w=(0,0)$、$b=0$。依次输入：

| 样本 | $x$ | $y$ |
| --- | --- | ---: |
| A | $(2,1)$ | $+1$ |
| B | $(-1,-2)$ | $-1$ |
| C | $(-2,1)$ | $-1$ |
| D | $(1,-2)$ | $+1$ |

对 A，初始得分为 0，满足 $y(w^\top x+b)\le0$，执行更新：

$$
w=(0,0)+1\times(+1)\times(2,1)=(2,1),\qquad b=0+1=1
$$

随后三例的得分分别为：

$$
f(B)=-3,\qquad f(C)=-2,\qquad f(D)=1
$$

它们的符号均与标签一致，不再更新。最终边界为：

$$
2x_1+x_2+1=0
$$

这个数据顺序只需一次更新便找到分离线；换一种样本顺序可能得到另一条同样能分开的边界，说明经典感知机不追求唯一解或最大间隔。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续变量、二元变量或已编码的离散变量。
- 因变量类型：二分类。
- 数据结构：二维特征矩阵。
- 是否适合高维数据：可以，但对噪声和尺度较敏感。
- 是否适合缺失较多数据：不适合，通常需先插补。
- 是否适合删失数据：不适合。
- 是否适合重复测量数据：基础感知机不直接处理相关结构。

### 5.2 示例表格

以简单感染初筛为例：

| Temperature | WBC | CRP | HeartRate | Infection |
| --- | --- | --- | --- | --- |
| 38.8 | 13.2 | 52 | 112 | 1 |
| 36.7 | 6.4 | 3 | 76 | 0 |
| 39.1 | 14.8 | 67 | 118 | 1 |
| 36.5 | 5.9 | 2 | 72 | 0 |

### 5.3 输入与产出

#### 输入

- 输入数据：样本特征矩阵和二分类标签。
- 关键变量：学习率、迭代轮数、是否加正则化。
- 需要预处理的内容：标准化、异常值检查、类别编码。

#### 产出

- 模型对象/统计结果：线性判别器参数。
- 参数估计：权重和偏置。
- 预测结果：二分类标签，部分实现也可返回决策函数值。
- 不确定性指标：通常依赖外部验证集准确率、召回率、AUC 等，而非参数置信区间。

## 6. 适用场景

- 适合：线性边界明显、需要简单快速基线模型的场景。
- 不适合：非线性关系复杂、类别边界高度重叠、需要概率输出和稳定校准的场景。
- 使用前需要特别检查的点：线性可分程度、特征尺度、类别不平衡。

## 7. 实现

### 7.1 Python

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

### 7.2 R

常用包：

- 无统一主流基础实现，实践中更常用 `keras` 或 `torch` 复现单层线性分类器

```r
perceptron_fit <- function(X, y, eta = 0.1, epochs = 1000) {
  y_signed <- ifelse(y == 1, 1, -1)
  w <- numeric(ncol(X))
  b <- 0

  for (epoch in seq_len(epochs)) {
    errors <- 0L
    for (i in seq_len(nrow(X))) {
      margin <- y_signed[i] * (sum(w * X[i, ]) + b)
      if (margin <= 0) {
        w <- w + eta * y_signed[i] * X[i, ]
        b <- b + eta * y_signed[i]
        errors <- errors + 1L
      }
    }
    if (errors == 0L) break
  }
  list(w = w, b = b, epochs = epoch)
}

X <- as.matrix(df[c("Temperature", "WBC", "CRP", "HeartRate")])
y <- df$Infection
fit <- perceptron_fit(scale(X), y)
score <- drop(scale(X) %*% fit$w + fit$b)
pred <- ifelse(score >= 0, 1, 0)
```

## 8. 结果如何解释

- 核心结果看什么：分类错误率、召回率、特异度，以及决策边界是否稳定。
- 每个主要参数如何解释：权重正负代表特征对分类方向的影响，但大小受量纲影响明显。
- 临床或医学意义如何表达：更适合作为简单筛查规则的技术基线，而不是最终解释模型。
- 常见误读：训练集完全分开不代表模型泛化好，也不代表真实机制就是线性的。

## 9. 假设诊断与稳健性

- **监控每轮误分类数**：线性可分时应最终归零；持续振荡提示类别重叠、异常值、噪声标签或学习率问题。
- **标准化特征**：否则大量纲变量主导更新步长；所有缩放参数必须只用训练折估计。
- **改变样本顺序与随机种子**：比较权重和验证性能，识别对训练顺序敏感的边界。
- **检查类别不平衡**：报告灵敏度、特异度和混淆矩阵，必要时重采样或采用代价敏感更新。
- **查看决策分数分布**：大量样本贴近 0 表明边界脆弱；感知机本身不提供概率或校准不确定性。
- **与线性基线比较**：同步评估 Logistic 回归和线性 SVM；若需要概率、正则化或最大间隔，它们通常更合适。
- **使用独立验证**：训练集收敛只证明被当前边界分开，不证明对新患者有效。

## 10. 推荐可视化

- 二维投影下的决策边界图。
- 误分类样本散点图。
- 类别分布与混淆矩阵。

## 11. 优势、局限与常见坑

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

## 12. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] 的区别：Logistic 回归输出概率并通过似然估计训练，感知机更强调误分类驱动的更新。
- 和 [[支持向量机（Support Vector Machine, SVM）]] 的区别：SVM 追求最大间隔，感知机只要求分开样本，不控制边界间隔。
- 和 [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]] 的区别：感知机是单层线性判别器，MLP 通过隐藏层学习非线性映射。

## 13. 医学研究中的典型应用

- 少量实验室指标的二分类初筛。
- 简单生理信号模式的线性判别。
- 作为更复杂模型前的可解释基线。

## 14. 关键术语

- **超平面**：由 $w^\top x+b=0$ 定义的线性决策边界。
- **决策分数**：$w^\top x+b$，其符号决定类别。
- **学习率**：每次误分类更新的步长 $\eta$。
- **在线学习**：样本逐个到达并即时更新模型的训练方式。
- **线性可分**：存在某个超平面能把两类训练样本完全分开。
- **感知机收敛定理**：线性可分且条件满足时，有限次误分类更新后算法收敛。
- **Pocket 策略**：非可分数据中保留迄今验证表现最佳的权重。
- **最大间隔**：在分离边界中最大化到最近样本的距离；经典感知机不优化它。

## 15. 相关方法

- [[Logistic回归（Logistic Regression）]]
- [[支持向量机（Support Vector Machine, SVM）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]

## 16. 参考资料

- Rosenblatt F. The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*. 1958;65(6):386-408.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- scikit-learn Developers. `sklearn.linear_model.Perceptron`. [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) （访问日期：2026-07-02）

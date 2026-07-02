---
title: Bagging算法
english_name: Bootstrap Aggregating
slug: bootstrap-aggregating
aliases: [bagging, bootstrap aggregating, "Bagging算法（Bootstrap Aggregating）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 重抽样, 树模型]
status: 已建
difficulty: basic
question_type: 降低高方差模型的预测波动
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ipred, caret]
---

# Bagging算法（Bootstrap Aggregating）

## 1. 方法概览

### 1.1 定义

Bagging 是 Bootstrap Aggregating 的缩写，是一种通过自助抽样生成多个训练子集、分别训练基学习器、再对预测结果投票或平均的集成学习方法。它主要用于降低不稳定模型的方差，典型代表是 [[随机森林（Random Forest）]]。

### 1.2 它主要解决什么问题

- 研究问题：如何让单个高方差模型的预测结果更稳定。
- 适用任务：二分类、多分类、连续结局预测。
- 常见医学场景：临床风险分类、疾病复发预测、生物标志物组合预测。

### 1.3 直觉理解

Bagging 像是让多个“看过不同训练样本版本”的模型独立判断，再综合它们的意见。若每个模型都会受样本扰动影响，平均或投票后这种偶然波动会被抵消，最终预测更稳。

## 2. 数学形式

### 2.1 核心公式

给定训练集 $D=\{(x_i,y_i)\}_{i=1}^{n}$，第 $b$ 个 bootstrap 子集记为 $D_b$，在其上训练基学习器 $h_b(x)$。分类任务的集成预测为：

$$
\hat y=\arg\max_k\sum_{b=1}^{B}I(h_b(x)=k)
$$

回归任务的集成预测为：

$$
\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}h_b(x)
$$

若基学习器方差为 $\sigma^2$，两两相关性近似为 $\rho$，则平均后的方差近似为：

$$
\mathrm{Var}(\hat f)=\sigma^2\left(\rho+\frac{1-\rho}{B}\right)
$$

### 2.2 参数或统计量含义

- $B$：基学习器数量。
- bootstrap 抽样：带放回抽样，每个子集大小通常等于原训练集大小。
- $h_b(x)$：第 $b$ 个基学习器。
- $\rho$：基学习器之间的预测相关性，越低越有利于集成。
- OOB 样本：某次 bootstrap 中未被抽中的样本，可用于袋外误差估计。

### 2.3 关键假设

- 基学习器对训练样本扰动较敏感，即本身方差较高。
- 多个基学习器的错误模式不完全相同。
- 主要目标是提升预测稳定性，而非获得简单参数解释。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可，取决于基学习器。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：可用，但需注意基学习器和样本量。
- 是否适合缺失较多数据：通常需先处理缺失。
- 是否适合删失数据：普通 Bagging 不直接处理删失结局。
- 是否适合重复测量数据：不直接适合，需要先处理相关结构或使用专门模型。

### 3.2 示例表格

以术后 30 天并发症预测为例：

| Age | ASA | SurgeryTime | Albumin | Emergency | Complication30d |
| --- | --- | --- | --- | --- | --- |
| 72 | 3 | 145 | 32 | 1 | 1 |
| 48 | 2 | 80 | 42 | 0 | 0 |
| 64 | 3 | 130 | 35 | 0 | 1 |
| 55 | 2 | 95 | 40 | 0 | 0 |
| 77 | 4 | 160 | 29 | 1 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：基学习器类型、基学习器数量、每次抽样比例、特征抽样比例。
- 需要预处理的内容：缺失处理、类别编码、训练测试集划分、类别不平衡处理。

#### 产出

- 模型对象/统计结果：Bagging 集成模型、袋外误差、各基学习器预测。
- 参数估计：通常不输出传统系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：OOB 误差、交叉验证指标、测试集 AUC / F1 / MSE。

## 4. 适用场景

- 适合：单模型不稳定、非线性关系明显、希望降低预测方差的任务。
- 不适合：基学习器本身偏差很高且变化很小的任务，例如简单线性模型的直接 Bagging 通常收益有限。
- 使用前需要特别检查的点：基模型是否足够多样、OOB 或外部验证表现、类别不平衡下的阈值选择。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("surgical_complication.csv")
X = df[["Age", "ASA", "SurgeryTime", "Albumin", "Emergency"]]
y = df["Complication30d"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fit = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, min_samples_leaf=10),
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
fit.fit(X_train, y_train)

print("OOB score:", fit.oob_score_)
pred_prob = fit.predict_proba(X_test)[:, 1]
```

### 5.2 R

常用包：

- `ipred`
- `caret`

```r
library(ipred)

fit <- bagging(
  Complication30d ~ Age + ASA + SurgeryTime + Albumin + Emergency,
  data = df_train,
  nbagg = 200,
  coob = TRUE
)

pred <- predict(fit, newdata = df_test, type = "prob")
```

## 6. 结果如何解释

- 核心结果看什么：测试集性能、OOB 误差、模型稳定性。
- 每个主要参数如何解释：树数越多通常越稳定；抽样比例影响基模型差异；基学习器复杂度影响偏差与方差。
- 临床或医学意义如何表达：Bagging 更适合作为稳健预测工具，不适合直接解释单个变量的独立效应。
- 常见误读：OOB 误差不能替代外部验证；投票多不代表模型已校准。

## 7. 推荐可视化

- OOB 误差随基学习器数量变化曲线。
- ROC 曲线、PR 曲线和校准曲线。
- 基模型预测分布或集成预测概率分布。

## 8. 优势、局限与常见坑

### 优势

- 能显著降低高方差模型的预测波动。
- 基学习器之间可并行训练。
- 通常调参压力小于 Boosting。

### 局限

- 对降低偏差帮助有限。
- 模型解释性弱于单个模型。
- 基模型数量多时预测成本增加。

### 常见坑

- 对稳定模型机械使用 Bagging，收益不明显。
- 忽略类别不平衡和概率校准。
- 只报告准确率，不报告医学预测中更关键的灵敏度、特异度、校准和临床阈值表现。

## 9. 与相近方法的区别

- 和 [[随机森林（Random Forest）]] 的区别：随机森林是 Bagging 的特化版本，额外在每个节点随机抽取特征。
- 和 [[Boosting算法（Boosting）]] 的区别：Bagging 并行训练多个模型以降方差；Boosting 串行训练模型以逐步纠错。
- 和 [[投票集成（Voting Ensemble）]] 的区别：投票集成通常组合已训练的不同模型；Bagging 通过 bootstrap 自动生成多个训练版本。

## 10. 医学研究中的典型应用

- 临床表格数据风险预测的稳健基线模型。
- 病理或组学特征较多时的分类任务。
- 小到中等样本量下对单棵树不稳定性的修正。

## 11. 相关方法

- [[随机森林（Random Forest）]]
- [[极端随机树（Extremely Randomized Trees）]]
- [[投票集成（Voting Ensemble）]]
- [[Boosting算法（Boosting）]]

## 12. 参考资料

- Breiman L. Bagging predictors. *Mach Learn*. 1996;24:123-140.
- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- scikit-learn Developers. `sklearn.ensemble.BaggingClassifier`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) （访问日期：2026-07-02）

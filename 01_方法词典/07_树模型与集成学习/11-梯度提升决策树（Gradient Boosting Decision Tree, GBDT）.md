---
title: 梯度提升决策树
english_name: Gradient Boosting Decision Tree, GBDT
slug: gradient-boosting-decision-tree-gbdt
aliases: [gradient boosting decision tree, gradient boosted decision trees, "梯度提升决策树（Gradient Boosting Decision Tree, GBDT）"]
category: 树模型与集成学习
subcategory: Boosting集成
tags: [医学统计, 数据科学, 集成学习, 提升方法, 树模型]
status: 已建
difficulty: intermediate
question_type: 梯度提升树建模
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [gbm]
---

# 梯度提升决策树（Gradient Boosting Decision Tree, GBDT）

## 1. 方法概览

### 1.1 定义

梯度提升决策树是以决策树为基学习器的梯度提升方法。它通过逐轮拟合损失函数的负梯度或伪残差，把多棵浅树叠加成一个强预测模型。

### 1.2 它主要解决什么问题

- 研究问题：如何用一系列浅层决策树逐步逼近复杂非线性预测函数。
- 适用任务：二分类、多分类、连续结局预测。
- 常见医学场景：临床风险预测、住院时长或费用预测、疾病复发和死亡风险建模。

### 1.3 直觉理解

GBDT 每一轮都问：“当前模型还错在哪里？”然后训练一棵新树去拟合这些错误方向。新树不是独立投票，而是作为对已有模型的一小步修正。

## 2. 数学形式

### 2.1 核心公式

GBDT 的加法模型为：

$$
F_M(x)=F_0(x)+\sum_{m=1}^{M}\nu h_m(x)
$$

其中 $h_m(x)$ 是第 $m$ 棵决策树，$\nu$ 是学习率。

初始化常数模型：

$$
F_0(x)=\arg\min_{\gamma}\sum_{i=1}^{n}L(y_i,\gamma)
$$

第 $m$ 轮计算负梯度，也称伪残差：

$$
r_{im}=-\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
$$

用决策树拟合 $\{(x_i,r_{im})\}$，再更新：

$$
F_m(x)=F_{m-1}(x)+\nu h_m(x)
$$

对于平方误差损失，伪残差就是普通残差：

$$
r_{im}=y_i-F_{m-1}(x_i)
$$

### 2.2 参数或统计量含义

- `n_estimators`：树的数量。
- `learning_rate`：每棵树对最终模型的贡献。
- `max_depth`：单棵树的最大深度。
- `subsample`：每轮训练使用的样本比例，形成随机梯度提升。
- 伪残差：每轮新树要拟合的目标。

### 2.3 关键假设

- 数据中存在树模型可捕捉的非线性和交互。
- 损失函数与任务目标匹配。
- 学习率、树深和树数需要共同控制，以避免过拟合。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可，类别变量通常需编码。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：适合中高维表格数据。
- 是否适合缺失较多数据：经典 scikit-learn GBDT 通常需先处理缺失。
- 是否适合删失数据：普通 GBDT 不直接适合。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

以慢病患者年度医疗费用预测为例：

| Age | Charlson | HbA1c | eGFR | PriorCost | AnnualCost |
| --- | --- | --- | --- | --- | --- |
| 71 | 4 | 8.9 | 48 | 18300 | 24600 |
| 52 | 1 | 6.7 | 92 | 4200 | 5100 |
| 63 | 3 | 7.8 | 55 | 11200 | 15100 |
| 44 | 0 | 6.1 | 101 | 1800 | 2300 |
| 68 | 2 | 7.1 | 60 | 9500 | 12800 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：损失函数、树数、学习率、树深、叶节点最小样本数、采样比例。
- 需要预处理的内容：缺失处理、类别编码、训练验证划分、异常值检查。

#### 产出

- 模型对象/统计结果：梯度提升树模型、训练/验证误差、特征重要性。
- 参数估计：不输出传统回归系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：交叉验证性能、测试集 AUC / MSE、校准指标。

## 4. 适用场景

- 适合：表格数据、非线性和交互明显、预测性能优先的任务。
- 不适合：样本极小、强可解释性或因果解释优先、缺失机制复杂但未处理的任务。
- 使用前需要特别检查的点：学习率和树数是否匹配、是否需要早停、是否过拟合、概率是否校准。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("annual_cost.csv")
X = df[["Age", "Charlson", "HbA1c", "eGFR", "PriorCost"]]
y = df["AnnualCost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

fit = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42
)
fit.fit(X_train, y_train)

y_pred = fit.predict(X_test)
```

### 5.2 R

常用包：

- `gbm`

```r
library(gbm)

fit <- gbm(
  AnnualCost ~ Age + Charlson + HbA1c + eGFR + PriorCost,
  data = df_train,
  distribution = "gaussian",
  n.trees = 300,
  interaction.depth = 3,
  shrinkage = 0.03,
  bag.fraction = 0.8
)

pred <- predict(fit, newdata = df_test, n.trees = 300)
```

## 6. 结果如何解释

- 核心结果看什么：验证集性能、学习曲线、特征重要性、残差分布或校准表现。
- 每个主要参数如何解释：学习率越小，每棵树贡献越小，通常需要更多树；树深决定可捕捉的交互阶数。
- 临床或医学意义如何表达：适合表达“模型捕捉到复杂非线性风险模式”，不适合直接解释为可干预原因。
- 常见误读：把 GBDT 的特征重要性当作变量独立效应或因果效应。

## 7. 推荐可视化

- 训练/验证损失曲线。
- 真实值 vs 预测值散点图或分类 ROC 曲线。
- 特征重要性图和部分依赖图。

## 8. 优势、局限与常见坑

### 优势

- 表格数据预测能力强。
- 能处理复杂非线性和交互。
- 损失函数灵活，可用于分类和回归。

### 局限

- 调参较多。
- 对异常值和噪声较敏感。
- 训练通常比随机森林更依赖验证策略。

### 常见坑

- 学习率过大且树过深导致过拟合。
- 只做内部交叉验证，不做外部验证。
- 忽略概率校准和临床决策阈值。

## 9. 与相近方法的区别

- 和 [[梯度提升回归（Gradient Boosting Regression）]] 的区别：本卡是 GBDT 总论；梯度提升回归是连续结局场景的具体变体。
- 和 [[XGBoost（Extreme Gradient Boosting, XGBoost）]] 的区别：XGBoost 在 GBDT 基础上加入二阶优化、显式正则化和工程优化。
- 和 [[LightGBM（Light Gradient Boosting Machine）]] 的区别：LightGBM 强调直方图分箱、leaf-wise 生长和大规模数据效率。

## 10. 医学研究中的典型应用

- 慢病管理中的费用、住院天数或连续风险评分预测。
- 住院死亡、再入院、感染、并发症等分类风险模型。
- 多变量临床表格数据中的非线性基线模型。

## 11. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[AdaBoost（Adaptive Boosting）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 12. 参考资料

- Friedman JH. Greedy function approximation: a gradient boosting machine. *Ann Stat*. 2001;29(5):1189-1232.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- scikit-learn Developers. `sklearn.ensemble.GradientBoostingRegressor`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) （访问日期：2026-07-02）

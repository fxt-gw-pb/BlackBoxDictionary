---
title: Boosting算法
english_name: Boosting
slug: boosting
aliases: [boosting, "Boosting算法（Boosting）"]
category: 树模型与集成学习
subcategory: Boosting集成
tags: [医学统计, 数据科学, 集成学习, 提升方法]
status: 已建
difficulty: intermediate
question_type: 串行纠错式预测建模
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [gbm, adabag]
---

# Boosting算法（Boosting）

## 1. 方法概览

### 1.1 定义

Boosting 是一类串行集成学习方法，它把多个弱学习器按顺序组合起来，每一轮都让新模型重点修正当前集成模型的错误。AdaBoost、GBDT、XGBoost 和 LightGBM 都属于 Boosting 家族。

### 1.2 它主要解决什么问题

- 研究问题：如何把多个略优于随机猜测的弱模型逐步组合成强预测模型。
- 适用任务：二分类、多分类、连续结局预测和排序任务。
- 常见医学场景：疾病风险预测、并发症预测、电子病历表格数据建模。

### 1.3 直觉理解

Boosting 像是连续改错：先用一个简单模型做初步预测，然后让后续模型专门学习前面没处理好的样本或残差。多轮之后，整体模型会比任何单个弱学习器更强。

## 2. 数学形式

### 2.1 核心公式

Boosting 通常写成加法模型：

$$
F_M(x)=\sum_{m=1}^{M}\alpha_m h_m(x)
$$

其中 $h_m(x)$ 是第 $m$ 个弱学习器，$\alpha_m$ 是其权重或步长。

以二分类 AdaBoost 为例，样本权重初始化为：

$$
w_i^{(1)}=\frac{1}{n}
$$

第 $m$ 轮弱分类器的加权误差为：

$$
\epsilon_m=\sum_{i=1}^{n}w_i^{(m)}I(h_m(x_i)\neq y_i)
$$

分类器权重为：

$$
\alpha_m=\frac{1}{2}\log\frac{1-\epsilon_m}{\epsilon_m}
$$

最终分类器为：

$$
H(x)=\mathrm{sign}\left(\sum_{m=1}^{M}\alpha_m h_m(x)\right)
$$

### 2.2 参数或统计量含义

- `n_estimators`：提升轮数或弱学习器数量。
- `learning_rate`：每轮更新幅度，也称 shrinkage。
- 弱学习器：通常是浅层决策树或树桩。
- 样本权重/负梯度：决定下一轮模型重点学习哪里。

### 2.3 关键假设

- 数据中存在弱学习器可逐步捕捉的结构。
- 标注错误和异常值不过多，否则后续模型可能反复追逐噪声。
- 需要通过验证集、早停或正则化控制过拟合。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可，类别变量常需编码。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：适合中高维表格数据，但需调参。
- 是否适合缺失较多数据：取决于具体实现，XGBoost/LightGBM 有原生缺失处理能力。
- 是否适合删失数据：普通 Boosting 不直接适合，需使用专门生存 Boosting。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

以败血症早期预警为例：

| Age | HR | SBP | Lactate | WBC | Sepsis |
| --- | --- | --- | --- | --- | --- |
| 76 | 118 | 88 | 3.4 | 16.2 | 1 |
| 52 | 82 | 126 | 1.1 | 7.8 | 0 |
| 69 | 105 | 94 | 2.6 | 12.4 | 1 |
| 41 | 74 | 132 | 0.9 | 6.1 | 0 |
| 63 | 96 | 108 | 1.8 | 10.2 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：目标变量和特征矩阵。
- 关键变量：弱学习器数量、学习率、树深、损失函数、采样和正则化参数。
- 需要预处理的内容：训练验证划分、类别编码、缺失机制判断、异常值检查。

#### 产出

- 模型对象/统计结果：Boosting 集成模型、训练/验证误差曲线、特征重要性。
- 参数估计：一般不输出传统可解释系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：交叉验证性能、测试集 AUC / PR-AUC / MSE、校准指标。

## 4. 适用场景

- 适合：复杂非线性、交互较多、追求预测性能的表格数据任务。
- 不适合：标签噪声很高、样本极小、强因果解释为主要目标的任务。
- 使用前需要特别检查的点：学习率与树数平衡、过拟合、异常值、类别不平衡、概率校准。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("sepsis_alert.csv")
X = df[["Age", "HR", "SBP", "Lactate", "WBC"]]
y = df["Sepsis"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fit = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
fit.fit(X_train, y_train)

pred_prob = fit.predict_proba(X_test)[:, 1]
```

### 5.2 R

常用包：

- `gbm`

```r
library(gbm)

fit <- gbm(
  Sepsis ~ Age + HR + SBP + Lactate + WBC,
  data = df_train,
  distribution = "bernoulli",
  n.trees = 200,
  interaction.depth = 3,
  shrinkage = 0.05
)

pred <- predict(fit, newdata = df_test, n.trees = 200, type = "response")
```

## 6. 结果如何解释

- 核心结果看什么：验证集性能、早停轮数、学习曲线、特征重要性、概率校准。
- 每个主要参数如何解释：学习率越小通常需要更多轮数；树越深越能捕捉交互但越容易过拟合。
- 临床或医学意义如何表达：适合表达模型区分高危个体的能力，不应把重要性排序直接解释为因果作用。
- 常见误读：训练误差持续下降不代表泛化性能持续提升。

## 7. 推荐可视化

- 训练集和验证集损失随迭代轮数变化图。
- ROC 曲线、PR 曲线和校准曲线。
- 特征重要性图或 SHAP 总结图。

## 8. 优势、局限与常见坑

### 优势

- 能逐步降低偏差，预测性能通常较强。
- 可灵活适配多种损失函数。
- 对非线性和交互建模能力强。

### 局限

- 对噪声和异常值较敏感。
- 调参空间较大。
- 训练过程串行，通常比 Bagging 更难并行。

### 常见坑

- 学习率过大、树太深或轮数过多导致过拟合。
- 只报告 AUC，忽略校准和临床阈值。
- 在医学研究中把模型解释图误写成因果发现。

## 9. 与相近方法的区别

- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：Bagging 并行降方差；Boosting 串行纠错，常能更强地降低偏差。
- 和 [[AdaBoost（Adaptive Boosting）]] 的区别：AdaBoost 是经典 Boosting 实现，显式更新样本权重。
- 和 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]] 的区别：GBDT 用负梯度统一描述每轮修正，是现代 Boosting 树模型的主线。

## 10. 医学研究中的典型应用

- ICU 患者死亡或恶化风险预测。
- 住院再入院、术后并发症、疾病复发等二分类预测。
- 电子病历表格数据中的高性能预测基线。

## 11. 相关方法

- [[AdaBoost（Adaptive Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 12. 参考资料

- Schapire RE. The strength of weak learnability. *Mach Learn*. 1990;5:197-227.
- Freund Y, Schapire RE. A decision-theoretic generalization of on-line learning and an application to boosting. *J Comput Syst Sci*. 1997;55(1):119-139.
- Friedman JH. Greedy function approximation: a gradient boosting machine. *Ann Stat*. 2001;29(5):1189-1232.

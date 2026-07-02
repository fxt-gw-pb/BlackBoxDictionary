---
title: AdaBoost
english_name: Adaptive Boosting
slug: adaptive-boosting
aliases: [adaboost, adaptive boosting, "AdaBoost（Adaptive Boosting）"]
category: 树模型与集成学习
subcategory: Boosting集成
tags: [医学统计, 数据科学, 集成学习, 提升方法, 分类]
status: 已建
difficulty: intermediate
question_type: 自适应加权分类
data_type: [表格数据]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn]
r_packages: [adabag, fastAdaboost]
---

# AdaBoost（Adaptive Boosting）

## 1. 方法概览

### 1.1 定义

AdaBoost 是 Adaptive Boosting 的缩写，是一种经典 Boosting 算法。它通过反复调整样本权重，让后续弱分类器更关注前一轮错分的样本，并把多个弱分类器加权组合成强分类器。

### 1.2 它主要解决什么问题

- 研究问题：如何把一组略优于随机猜测的弱分类器组合为强分类器。
- 适用任务：二分类和多分类分类任务。
- 常见医学场景：疾病有无判别、筛查阳性预测、欺诈或异常事件识别。

### 1.3 直觉理解

AdaBoost 每一轮都会“放大错题”。前一轮被分错的样本在下一轮获得更高权重，因此新弱分类器会更努力处理这些难分样本。最终预测是多个弱分类器的加权投票。

## 2. 数学形式

### 2.1 核心公式

设 $y_i\in\{-1,+1\}$，初始样本权重为：

$$
D_1(i)=\frac{1}{n}
$$

第 $t$ 轮训练弱分类器 $h_t(x)$，加权错误率为：

$$
\epsilon_t=\sum_{i=1}^{n}D_t(i)I(h_t(x_i)\neq y_i)
$$

弱分类器权重为：

$$
\alpha_t=\frac{1}{2}\log\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$

样本权重更新为：

$$
D_{t+1}(i)=\frac{D_t(i)\exp[-\alpha_t y_i h_t(x_i)]}{Z_t}
$$

最终分类器为：

$$
H(x)=\mathrm{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)
$$

### 2.2 参数或统计量含义

- $D_t(i)$：第 $t$ 轮样本 $i$ 的权重。
- $\epsilon_t$：第 $t$ 个弱分类器的加权错误率。
- $\alpha_t$：弱分类器权重，错误率越低权重越高。
- $Z_t$：归一化因子。
- `n_estimators`：弱分类器数量。
- `learning_rate`：每个弱分类器权重的缩放系数。

### 2.3 关键假设

- 弱学习器在加权数据上能达到优于随机猜测的性能。
- 标签噪声不宜过高，否则算法会持续追逐错误标签。
- 分类任务目标和损失设定适合加权投票框架。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可。
- 因变量类型：二分类或多分类。
- 数据结构：监督学习宽表数据。
- 是否适合高维数据：可用，但需控制弱学习器复杂度。
- 是否适合缺失较多数据：通常需先处理缺失。
- 是否适合删失数据：不直接适合。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

以急诊患者是否需要 ICU 收治为例：

| Age | RR | SBP | Lactate | GCS | ICU |
| --- | --- | --- | --- | --- | --- |
| 81 | 28 | 86 | 4.1 | 12 | 1 |
| 36 | 16 | 122 | 1.0 | 15 | 0 |
| 67 | 24 | 95 | 2.8 | 14 | 1 |
| 45 | 18 | 118 | 1.3 | 15 | 0 |
| 72 | 26 | 90 | 3.2 | 13 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和分类标签。
- 关键变量：弱分类器、弱分类器数量、学习率、树桩或浅树深度。
- 需要预处理的内容：缺失处理、异常值检查、类别不平衡处理、训练测试划分。

#### 产出

- 模型对象/统计结果：AdaBoost 分类器、弱分类器权重、特征重要性。
- 参数估计：不输出传统回归系数。
- 预测结果：类别和分类概率或决策分数。
- 不确定性指标：测试集 AUC / F1、校准曲线、交叉验证性能。

## 4. 适用场景

- 适合：较干净的分类数据、需要从简单弱分类器逐步增强的任务。
- 不适合：标签错误多、异常值多、强类别不平衡但未做处理的任务。
- 使用前需要特别检查的点：错分样本是否主要是噪声、弱学习器复杂度、学习率与轮数。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("emergency_icu.csv")
X = df[["Age", "RR", "SBP", "Lactate", "GCS"]]
y = df["ICU"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fit = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
fit.fit(X_train, y_train)

pred_prob = fit.predict_proba(X_test)[:, 1]
```

### 5.2 R

常用包：

- `adabag`

```r
library(adabag)

fit <- boosting(
  ICU ~ Age + RR + SBP + Lactate + GCS,
  data = df_train,
  boos = TRUE,
  mfinal = 100
)

pred <- predict.boosting(fit, newdata = df_test)
```

## 6. 结果如何解释

- 核心结果看什么：测试集判别性能、混淆矩阵、错误样本分布、校准情况。
- 每个主要参数如何解释：弱分类器越复杂越容易过拟合；轮数越多越可能追逐噪声。
- 临床或医学意义如何表达：可用于构建分类预测器，但重要性和错分权重不等于病因解释。
- 常见误读：AdaBoost 关注错分样本，不代表这些样本一定具有临床特殊机制，也可能只是标签错误或异常值。

## 7. 推荐可视化

- 训练和验证错误率随轮数变化曲线。
- 混淆矩阵、ROC 曲线、PR 曲线。
- 特征重要性条形图。

## 8. 优势、局限与常见坑

### 优势

- 理论清晰，实现简单。
- 能把简单弱分类器组合成强分类器。
- 对特征缩放通常不敏感。

### 局限

- 对噪声和异常值较敏感。
- 分类概率可能需要校准。
- 在复杂表格任务上常被 XGBoost、LightGBM 等现代 GBDT 实现替代。

### 常见坑

- 使用太深的基树，让每轮弱学习器不再“弱”。
- 标签噪声未处理，导致模型持续放大错误样本。
- 只看准确率，忽略少数类召回率和临床成本。

## 9. 与相近方法的区别

- 和 [[Boosting算法（Boosting）]] 的区别：AdaBoost 是 Boosting 的经典实例，核心是样本权重自适应更新。
- 和 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]] 的区别：GBDT 用负梯度统一处理多种损失；AdaBoost 以指数损失和样本权重更新为代表。
- 和 [[XGBoost（Extreme Gradient Boosting, XGBoost）]] 的区别：XGBoost 是正则化的二阶梯度提升树，更适合复杂表格任务。

## 10. 医学研究中的典型应用

- 急诊或 ICU 二分类风险筛查。
- 影像组学二分类模型的轻量基线。
- 标注较干净的疾病有无判别任务。

## 11. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 12. 参考资料

- Freund Y, Schapire RE. A decision-theoretic generalization of on-line learning and an application to boosting. *J Comput Syst Sci*. 1997;55(1):119-139.
- Hastie T, Rosset S, Zhu J, Zou H. Multi-class AdaBoost. *Stat Interface*. 2009;2(3):349-360.
- scikit-learn Developers. `sklearn.ensemble.AdaBoostClassifier`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) （访问日期：2026-07-02）

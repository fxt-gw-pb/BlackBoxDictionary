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

### 1.1 一句话本质

AdaBoost 通过提高错分样本的权重，让后一轮弱分类器更关注“难题”，最后按弱分类器质量加权投票。

### 1.2 定义

AdaBoost 是 Adaptive Boosting 的缩写，是 [[Boosting算法（Boosting）]] 的经典实现。二分类版本通常用树桩作为弱学习器，逐轮更新样本权重，并以错误率决定每个弱分类器在最终决策中的权重。

### 1.3 它主要解决什么问题

- 把略优于随机猜测的弱分类器组合成强分类器。
- 用自适应样本权重聚焦当前决策边界附近的困难病例。
- 在不训练深树的情况下构造非线性分类边界。

### 1.4 直觉与类比

像复习错题：答对的题暂时降低优先级，答错的题在下一轮出现得更频繁。表现好的辅导老师在最后投票时权重也更高。

## 2. 核心思想与原理

### 2.1 两层加权

AdaBoost 同时调整两类权重：错分病例的训练权重上升；错误率较低的弱分类器在最终投票中的权重更大。

### 2.2 与指数损失的关系

其逐轮更新可理解为前向分步最小化指数损失：

$$
L(y,F)=\exp[-yF(x)]
$$

当病例被错分时，$yF(x)$ 为负，损失迅速增大，因此模型会强烈关注它。这也解释了 AdaBoost 对错误标签和离群点较敏感。

### 2.3 多分类说明

经典推导以 $y\in\{-1,+1\}$ 的二分类为主。多分类可使用 SAMME 等推广；软件实现细节可能不同，分析时应说明算法版本和基学习器。

## 3. 数学形式

### 3.1 加权误差与分类器权重

初始化 $D_1(i)=1/n$。第 $t$ 轮弱分类器的加权错误率为：

$$
\epsilon_t=\sum_{i=1}^{n}D_t(i)I\{h_t(x_i)\ne y_i\}
$$

$$
\alpha_t=\frac12\log\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$

若 $\epsilon_t\lt0.5$，则 $\alpha_t\gt0$；错误率越低，投票权越大。

### 3.2 样本权重更新

$$
D_{t+1}(i)=
\frac{D_t(i)\exp[-\alpha_ty_ih_t(x_i)]}{Z_t}
$$

正确分类时指数因子为 $\exp(-\alpha_t)$，错分时为 $\exp(\alpha_t)$。$Z_t$ 使新权重之和为 1。

### 3.3 最终分类

$$
H(x)=\operatorname{sign}
\left[\sum_{t=1}^{T}\alpha_th_t(x)\right]
$$

### 3.4 关键条件

| 条件 | 违反后果 | 如何检查 |
| --- | --- | --- |
| 每轮弱分类器优于随机猜测 | $\alpha_t$ 无正贡献 | 查看逐轮错误率 |
| 标签噪声较低 | 错标签权重不断增大 | 审核高权重病例 |
| 基学习器足够简单 | 单轮过拟合，集成收益变小 | 比较树桩与浅树 |
| 训练数据代表目标人群 | 外部性能下降 | 外部或时间验证 |

## 4. 手把手算例

有 4 名患者，初始权重均为 $0.25$。第一棵树桩分对 3 人、分错 1 人。

**Step 1：加权错误率。**

$$
\epsilon_1=0.25
$$

**Step 2：计算树桩权重。**

$$
\alpha_1=\frac12\log\left(\frac{0.75}{0.25}\right)
=\frac12\log 3\approx0.549
$$

**Step 3：更新未归一化样本权重。**

每个正确病例：

$$
0.25\exp(-0.549)\approx0.144
$$

错分病例：

$$
0.25\exp(0.549)\approx0.433
$$

总和约为 $0.866$。归一化后，3 个正确病例各占：

$$
0.144/0.866\approx0.167
$$

错分病例占：

$$
0.433/0.866\approx0.500
$$

**结论：** 下一棵树训练时，那个错分病例独自占一半权重；若它其实是录入错误，算法也会同样执着地追逐它。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 结局为二分类或多分类。
- 特征可为连续或编码后的类别变量。
- 基学习器必须支持样本权重。
- 通常需预先处理缺失；极端类别不平衡时需结合合适指标与阈值。

### 5.2 产出

模型输出类别、决策分数和实现所定义的概率估计，还可查看各弱分类器权重与错误率。概率是否校准必须另行验证。

## 6. 适用场景

- 适合：标签较干净、样本量中等、希望用浅树获得非线性分类器。
- 慎用：错标签多、离群病例多、类别极不平衡且漏诊成本高。
- 在医学研究中，最好把它作为候选预测模型，而非病因解释工具。

## 7. 实现

### 7.1 Python

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=200,
    learning_rate=0.05,
    random_state=42,
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, prob))
print("Brier:", brier_score_loss(y_test, prob))
print(model.estimator_errors_[:5])
```

### 7.2 R

```r
library(adabag)

fit <- boosting(
  ICU ~ Age + RR + SBP + Lactate + GCS,
  data = train,
  boos = TRUE,
  mfinal = 200,
  coeflearn = "Breiman"
)

pred <- predict.boosting(fit, newdata = test)
prob <- pred$prob[, 2]
table(observed = test$ICU, predicted = pred$class)
```

## 8. 结果如何解释

- 决策分数是弱分类器加权和；绝对值越大通常表示分类间隔越大。
- `estimator_errors_` 高说明该轮弱学习器表现有限。
- 变量重要性只表示预测过程中被使用的程度。
- 临床报告应同时给出敏感度、特异度、PR-AUC、校准与阈值依据。

## 9. 诊断与稳健性

1. 画训练和验证错误率随轮数变化。
2. 列出最终高权重病例，核查标签、缺失和异常测量。
3. 比较树桩与深度 2 或 3 的浅树。
4. 在类别不平衡时重点检查 PR 曲线和少数类召回率。
5. 用时间切分或外部中心检验泛化，并在需要时进行概率校准。

## 10. 推荐可视化

- 逐轮训练/验证误差曲线。
- 样本权重分布或高权重病例表。
- ROC、PR、校准曲线与混淆矩阵。
- 前若干特征的重要性图。

## 11. 优势、局限与常见坑

**优势：** 原理清晰、预处理要求较少、能把树桩组合成复杂边界。

**局限：** 对错标签和离群点敏感，训练串行，现代复杂表格任务中常不及正则化提升树。

**常见坑：** 使用过深基树；只调树数不调学习率；把决策分数直接当可靠概率；对高权重病例不做数据核查。

## 12. 与相近方法的区别

- [[Boosting算法（Boosting）]]：AdaBoost 是其中以样本重加权为核心的具体算法。
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]：后者直接拟合损失负梯度，可适配更多损失。
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]：后者使用二阶信息、显式正则化和采样。
- [[随机森林（Random Forest）]]：后者独立训练多树并投票，不逐轮追踪错分样本。

## 13. 医学研究中的典型应用

- 急诊转 ICU、筛查阳性与疾病有无判别。
- 影像组学或实验室指标的二分类基线。
- 可穿戴设备事件识别与异常信号分类。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| 树桩 | 深度为 1、只有一次分裂的决策树 |
| 样本权重 | 病例对下一轮训练误差的贡献 |
| 分类器权重 | 弱分类器在最终投票中的贡献 |
| 指数损失 | 对负分类间隔增长很快的损失 |
| 分类间隔 | $yF(x)$，正值表示分类方向正确 |

## 15. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 16. 参考资料

- Freund Y, Schapire RE. A decision-theoretic generalization of on-line learning and an application to boosting. *J Comput Syst Sci*. 1997;55(1):119-139.
- Hastie T, Rosset S, Zhu J, Zou H. Multi-class AdaBoost. *Stat Interface*. 2009;2(3):349-360.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- scikit-learn Developers. `AdaBoostClassifier` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) （访问日期：2026-07-09）

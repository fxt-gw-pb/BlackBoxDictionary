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

### 1.1 一句话本质

GBDT 把“拟合残差”推广为“拟合损失函数的负梯度”，让浅决策树在函数空间中逐步走向更低损失。

### 1.2 定义

梯度提升决策树是以回归树为基学习器的梯度提升方法。每轮计算当前预测的伪残差，用一棵新树拟合它，再把新树乘以学习率后加入现有模型。

### 1.3 它主要解决什么问题

- 用统一框架处理回归、二分类和多分类。
- 捕捉表格数据中的阈值、非线性和变量交互。
- 通过多轮浅树修正降低单棵浅树的偏差。

### 1.4 直觉与类比

若把损失看作山坡高度，当前模型站在山坡上。负梯度指出下降最快的方向，新树负责近似这个方向，学习率决定每一步走多远。

## 2. 核心思想与原理

### 2.1 从数值参数到函数空间

普通梯度下降更新参数向量；GBDT 更新的是预测函数 $F(x)$。因为任意方向很难直接表示，所以用一棵浅树去近似每轮最需要的函数修正。

### 2.2 伪残差

平方损失的负梯度正好是 $y-\hat y$，所以看起来像拟合残差。对二项偏差、绝对误差等损失，负梯度不再是普通残差，因此“拟合负梯度”是更准确的总括。

### 2.3 正则化方式

小学习率、浅树、叶节点最小样本数、子采样和早停共同决定有效复杂度。树数本身不是孤立的复杂度指标。

## 3. 数学形式

### 3.1 初始化

$$
F_0(x)=
\operatorname*{arg\,min}_{c}
\sum_{i=1}^{n}L(y_i,c)
$$

### 3.2 伪残差与更新

$$
r_{im}=
-\left.
\frac{\partial L[y_i,F(x_i)]}{\partial F(x_i)}
\right|_{F=F_{m-1}}
$$

用回归树拟合 $(x_i,r_{im})$。若第 $m$ 棵树有叶区域 $R_{jm}$，可在各叶内求步长：

$$
\gamma_{jm}=
\operatorname*{arg\,min}_{\gamma}
\sum_{x_i\in R_{jm}}
L[y_i,F_{m-1}(x_i)+\gamma]
$$

$$
F_m(x)=F_{m-1}(x)+
\nu\sum_j\gamma_{jm}I(x\in R_{jm})
$$

### 3.3 参数含义

- $\nu$：学习率。
- $M$：最大树数。
- 树深或叶节点数：单轮可表达的交互复杂度。
- `subsample`：每轮抽取的样本比例。
- 损失函数：定义优化目标与异常值敏感性。

### 3.4 关键条件

| 条件 | 违反后果 | 检查 |
| --- | --- | --- |
| 损失匹配研究目标 | 优化指标与临床目标脱节 | 同时报告多类指标 |
| 负梯度可被浅树近似 | 多轮改善缓慢 | 调整树深与特征 |
| 早停数据独立于测试集 | 测试性能乐观 | 固定训练/验证/测试三部分 |
| 树复杂度受控 | 训练集过拟合 | 学习曲线与外部验证 |

## 4. 手把手算例

考虑 4 名患者的二分类结局 $y=(0,0,1,1)$。使用 logistic 损失，模型输出 logit $F$，概率为 $p=1/(1+\exp(-F))$。

**Step 1：初始化。** 阳性率为 $0.5$，故初始 logit 为：

$$
F_0=\log\left(\frac{0.5}{0.5}\right)=0
$$

所有人的初始概率均为 $0.5$，平均 log loss 为 $-\log(0.5)=0.693$。

**Step 2：计算负梯度。** 二分类 logistic 损失对 logit 的负梯度为：

$$
r_i=y_i-p_i
$$

所以 $r=(-0.5,-0.5,0.5,0.5)$。

**Step 3：拟合一棵树。** 假设某特征把前两人与后两人分开，新树输出 $h=(-0.5,-0.5,0.5,0.5)$。取 $\nu=0.4$：

$$
F_1=0+0.4h=(-0.2,-0.2,0.2,0.2)
$$

新概率约为 $(0.450,0.450,0.550,0.550)$。

**Step 4：看损失。** 每名患者对真实类别的预测概率均约为 $0.550$：

$$
\operatorname{LogLoss}_1=-\log(0.550)\approx0.598
$$

**结论：** 新树拟合的不是最终标签，而是当前模型应向哪个方向移动。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 适用于连续、二分类和多分类结局。
- 经典实现通常要求类别特征编码并提前处理缺失。
- 不直接建模删失、聚类或重复测量相关性，需采用专用损失或扩展。

### 5.2 产出

输出预测值或概率、逐轮损失、树集成及特征重要性。它不直接提供传统回归系数与标准误。

## 6. 适用场景

- 表格数据中非线性和交互明显。
- 需要比单棵树更高性能，又希望用浅树控制复杂度。
- 不适合以变量效应估计或因果解释为首要目的的研究。

## 7. 实现

### 7.1 Python

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

model = GradientBoostingClassifier(
    loss="log_loss",
    n_estimators=600,
    learning_rate=0.03,
    max_depth=2,
    min_samples_leaf=20,
    subsample=0.8,
    n_iter_no_change=30,
    validation_fraction=0.2,
    random_state=42,
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob))
print(brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(gbm)

fit <- gbm(
  Death30d ~ Age + SOFA + Lactate + Creatinine,
  data = train,
  distribution = "bernoulli",
  n.trees = 600,
  interaction.depth = 2,
  n.minobsinnode = 20,
  shrinkage = 0.03,
  bag.fraction = 0.8,
  cv.folds = 5
)

best_n <- gbm.perf(fit, method = "cv", plot.it = FALSE)
prob <- predict(fit, test, n.trees = best_n, type = "response")
```

## 8. 结果如何解释

- 最佳迭代轮数是验证损失最低附近的轮数，不是树越多越好。
- 树深为 1 主要表达加性阈值效应；更深的树可表达更高阶交互。
- 预测概率需要在独立数据上检查校准。
- 重要性、部分依赖和 SHAP 描述模型行为，不自动等于生物机制。

## 9. 诊断与稳健性

1. 比较训练与验证偏差曲线。
2. 检查不同学习率和树数的组合，而非单独调一个参数。
3. 用重复交叉验证评估调参不稳定性。
4. 对分类模型检查 ROC、PR、Brier、校准斜率和决策曲线。
5. 进行时间外、中心外及关键亚组验证。

## 10. 推荐可视化

- 训练/验证损失随迭代变化图。
- ROC、PR、校准与决策曲线。
- 回归任务的真实值-预测值和残差图。
- permutation importance、部分依赖或 SHAP 图。

## 11. 优势、局限与常见坑

**优势：** 损失函数统一、预测能力强、自动学习非线性与交互。

**局限：** 调参较多、串行训练、对噪声敏感、参数解释不直观。

**常见坑：** 把伪残差一律称为普通残差；用测试集选轮数；学习率过大且树过深；忽视校准。

## 12. 与相近方法的区别

- [[梯度提升回归（Gradient Boosting Regression）]]：GBDT 是总框架，后者专门针对连续结局。
- [[AdaBoost（Adaptive Boosting）]]：AdaBoost 以样本权重与指数损失为代表。
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]：加入二阶近似、显式树正则化和工程优化。
- [[LightGBM（Light Gradient Boosting Machine）]]：以直方图、leaf-wise 等策略提高大规模训练效率。

## 13. 医学研究中的典型应用

- 死亡、感染、再入院和并发症风险预测。
- 费用、住院时长和连续风险评分预测。
- 临床、实验室、影像组学特征的非线性融合。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| 函数空间梯度下降 | 直接逐步更新预测函数 |
| 伪残差 | 当前损失关于预测的负梯度 |
| shrinkage | 用学习率缩小每棵树贡献 |
| stochastic boosting | 每轮只用一部分样本拟合 |
| deviance | 分类中常用的负对数似然型损失 |

## 15. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[AdaBoost（Adaptive Boosting）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 16. 参考资料

- Friedman JH. Greedy function approximation: a gradient boosting machine. *Ann Stat*. 2001;29(5):1189-1232.
- Friedman JH. Stochastic gradient boosting. *Comput Stat Data Anal*. 2002;38(4):367-378.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- scikit-learn Developers. Gradient Boosting User Guide. [https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) （访问日期：2026-07-09）

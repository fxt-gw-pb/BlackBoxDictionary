---
title: XGBoost
english_name: Extreme Gradient Boosting, XGBoost
slug: extreme-gradient-boosting-xgboost
aliases: [xgboost, extreme gradient boosting, "XGBoost（Extreme Gradient Boosting, XGBoost）"]
category: 树模型与集成学习
subcategory: Boosting集成
tags: [医学统计, 数据科学, 集成学习, 提升方法, 树模型]
status: 已建
difficulty: intermediate
question_type: 高性能梯度提升建模
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [xgboost, scikit-learn]
r_packages: [xgboost]
---

# XGBoost（Extreme Gradient Boosting, XGBoost）

## 1. 方法概览

### 1.1 一句话本质

XGBoost 是带二阶近似、显式树复杂度惩罚、采样与高效实现的梯度提升树。

### 1.2 定义

XGBoost 是 Extreme Gradient Boosting 的缩写。它延续 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]] 的逐轮加树思路，并利用损失的一阶和二阶导数评估叶节点与分裂，同时加入叶节点权重和叶数惩罚。

### 1.3 它主要解决什么问题

- 在复杂表格数据上获得强预测性能。
- 用正则化控制提升树复杂度。
- 高效处理稀疏输入、缺失值、行列采样和大规模训练。

### 1.4 直觉与类比

普通梯度提升知道下坡方向；XGBoost 还利用坡面的弯曲程度判断步幅，并对“修太多枝叶”收费，因此每次分裂都要证明收益足以抵消复杂度成本。

## 2. 核心思想与原理

### 2.1 二阶近似

每轮不直接反复求解原始目标，而是在当前预测附近做二阶泰勒展开。样本的一阶梯度 $g_i$ 表示下降方向，二阶梯度 $h_i$ 表示局部曲率。

### 2.2 显式树正则化

叶节点过多会增加 $\gamma T$，叶节点预测绝对值过大会增加 $\lambda\sum w_j^2/2$。因此分裂只有在损失下降足够大时才被接受。

### 2.3 缺失与采样

训练时可为缺失值学习默认分支方向；`subsample` 和 `colsample_bytree` 分别进行行、列采样。原生缺失处理并不等于理解了缺失机制，临床研究仍需检查缺失是否携带流程信息或造成偏倚。

## 3. 数学形式

### 3.1 正则化目标

$$
\mathcal L=
\sum_{i=1}^{n}l(y_i,\hat y_i)
+\sum_{t=1}^{K}\Omega(f_t)
$$

$$
\Omega(f)=\gamma T+\frac12\lambda\sum_{j=1}^{T}w_j^2
$$

### 3.2 第 $t$ 轮二阶近似

$$
\widetilde{\mathcal L}^{(t)}=
\sum_{i=1}^{n}
\left[g_if_t(x_i)+\frac12h_if_t^2(x_i)\right]
+\Omega(f_t)
$$

对叶节点 $j$，令 $G_j=\sum_{i\in I_j}g_i$、$H_j=\sum_{i\in I_j}h_i$，最优叶权重为：

$$
w_j^*=-\frac{G_j}{H_j+\lambda}
$$

### 3.3 分裂增益

$$
\operatorname{Gain}=
\frac12\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma
$$

增益为正才说明该分裂在正则化后值得保留。

### 3.4 关键参数

- `learning_rate`、`n_estimators`：步长与轮数。
- `max_depth`、`min_child_weight`、`gamma`：树结构约束。
- `subsample`、`colsample_bytree`：行列采样。
- `reg_alpha`、`reg_lambda`：L1 与 L2 正则。

## 4. 手把手算例

采用平方损失 $l=(\hat y-y)^2/2$。当前预测均为 0，4 个样本真实值为 $(1,2,4,5)$，因此：

$$
g=(-1,-2,-4,-5),\qquad h=(1,1,1,1)
$$

取 $\lambda=1$、$\gamma=0.1$。候选分裂把前两人与后两人分开。

**Step 1：汇总梯度。**

$$
G_L=-3,\ H_L=2,\qquad G_R=-9,\ H_R=2
$$

父节点为 $G=-12,H=4$。

**Step 2：计算分裂增益。**

$$
\begin{aligned}
\operatorname{Gain}
&=\frac12\left(
\frac{9}{3}+\frac{81}{3}-\frac{144}{5}
\right)-0.1\\
&=\frac12(3+27-28.8)-0.1\\
&=0.5
\end{aligned}
$$

增益为正，分裂值得保留。

**Step 3：计算两个叶节点值。**

$$
w_L^*=-\frac{-3}{2+1}=1,\qquad
w_R^*=-\frac{-9}{2+1}=3
$$

若学习率为 $0.1$，本轮实际给两组预测分别增加 $0.1$ 与 $0.3$。

**结论：** XGBoost 同时用梯度、曲率和正则化判断“是否分裂”以及“叶节点修正多少”。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续、二分类、多分类和排序任务均有对应目标函数。
- 输入通常为数值矩阵或稀疏矩阵；类别变量需按所用接口妥善处理。
- 可原生学习缺失默认方向，但仍须分析缺失机制与数据泄漏。
- 重复测量应按患者分组切分；删失结局需选择适当生存目标并明确估计对象。

### 5.2 产出

输出类别、概率、连续预测或排序分数，以及迭代记录和多种重要性。传统回归系数、标准误与因果效应不是其默认输出。

## 6. 适用场景

- 中大型表格数据，变量关系复杂且预测性能优先。
- 稀疏特征、缺失较多或需细致正则化的任务。
- 慎用于样本很小、外推要求高、强参数解释或因果推断任务。

## 7. 实现

### 7.1 Python

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

prob = model.predict_proba(X_test)[:, 1]
print(model.best_iteration)
print(roc_auc_score(y_test, prob))
print(brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(xgboost)

dtrain <- xgb.DMatrix(as.matrix(X_train), label = y_train)
dvalid <- xgb.DMatrix(as.matrix(X_valid), label = y_valid)
dtest <- xgb.DMatrix(as.matrix(X_test))

fit <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = 0.03,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain,
  nrounds = 2000,
  evals = list(valid = dvalid),
  early_stopping_rounds = 50,
  verbose = 0
)

prob <- predict(fit, dtest)
```

## 8. 结果如何解释

- `best_iteration` 是验证集早停选出的轮数，最终测试集不参与选择。
- gain、cover、weight 等重要性定义不同，报告时应注明类型。
- SHAP 值解释模型在给定数据分布下如何形成预测，不自动代表因果贡献。
- 分类概率须在独立数据上检查校准，必要时再校准。

## 9. 诊断与稳健性

1. 保存训练和验证指标，检查早停点。
2. 先联合约束树深、叶节点最小 Hessian 与学习率，再调采样和正则。
3. 比较随机种子和交叉验证折之间的性能及重要性稳定性。
4. 审查缺失率、缺失默认方向和可能的数据流程泄漏。
5. 做外部验证、亚组公平性、校准与决策曲线分析。

## 10. 推荐可视化

- 迭代轮数与训练/验证损失。
- ROC、PR、校准和决策曲线。
- gain 或 permutation importance 图。
- SHAP 总结图与局部解释图。

下图展示真实值与预测值的对照关系：

![](../../04_示例图像/xgboost_actual_vs_pred_optimized.png)

## 11. 优势、局限与常见坑

**优势：** 表格性能强，正则化完整，支持稀疏和缺失输入，训练工具成熟。

**局限：** 超参数多、解释复杂、概率可能失准、分布漂移时性能会下降。

**常见坑：** 用测试集早停；未设置验证集却声称早停；把默认重要性当因果；调参搜索后不做嵌套或独立评估。

## 12. 与相近方法的区别

- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]：XGBoost 增加二阶近似与显式结构正则。
- [[LightGBM（Light Gradient Boosting Machine）]]：两者目标相近，后者更强调直方图、leaf-wise、GOSS 与 EFB。
- [[随机森林（Random Forest）]]：随机森林并行平均，XGBoost 串行优化损失。
- [[AdaBoost（Adaptive Boosting）]]：AdaBoost 主要通过错分样本重加权。

## 13. 医学研究中的典型应用

- ICU 死亡、败血症、再入院和术后并发症预测。
- 临床变量与影像组学、组学特征融合。
- 医疗费用、住院时长和连续生物标志物预测。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| gradient | 损失对当前预测的一阶导数 |
| Hessian | 损失对当前预测的二阶导数 |
| gain | 分裂带来的正则化目标改善 |
| cover | 节点内二阶梯度总量等覆盖度量 |
| default direction | 特征缺失时自动采用的分支方向 |

## 15. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[LightGBM（Light Gradient Boosting Machine）]]
- [[随机森林（Random Forest）]]

## 16. 参考资料

- Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of KDD 2016*. 2016:785-794.
- XGBoost Contributors. XGBoost Documentation. [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/) （访问日期：2026-07-09）
- Mitchell R, Frank E. Accelerating the XGBoost algorithm using GPU computing. *PeerJ Comput Sci*. 2017;3:e127.

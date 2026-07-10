---
title: LightGBM
english_name: Light Gradient Boosting Machine
slug: light-gradient-boosting-machine
aliases: [lightgbm, lgbm, "LightGBM（Light Gradient Boosting Machine）"]
category: 树模型与集成学习
subcategory: Boosting集成
tags: [医学统计, 数据科学, 集成学习, 提升方法, 树模型]
status: 已建
difficulty: intermediate
question_type: 高效梯度提升建模
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [lightgbm, scikit-learn]
r_packages: [lightgbm]
---

# LightGBM（Light Gradient Boosting Machine）

## 1. 方法概览

### 1.1 一句话本质

LightGBM 用直方图近似分裂搜索，并优先分裂全树中收益最大的叶节点，以较低内存和较高速度训练梯度提升树。

### 1.2 定义

LightGBM 是高效的梯度提升树实现。其代表性设计包括 histogram 分箱、leaf-wise 生长、Gradient-based One-Side Sampling（GOSS）和 Exclusive Feature Bundling（EFB）。

### 1.3 它主要解决什么问题

- 大样本、高维或稀疏表格数据的训练效率。
- 频繁调参和多轮提升带来的时间与内存成本。
- 在保持强预测能力时减少精确阈值枚举。

### 1.4 直觉与类比

它先把连续数值放入有限个刻度桶，不再逐个尝试所有原始阈值；长树时也不要求同一层一起扩展，而是每次优先处理当前最能降低损失的叶子。

## 2. 核心思想与原理

### 2.1 直方图算法

连续特征被离散到有限 bins。每个 bin 累加样本数、梯度和 Hessian，候选分裂只需扫描 bin 汇总量。这样速度和内存受 bin 数而非原始不同取值数主导。

### 2.2 Leaf-wise 生长

传统 level-wise 方法同层扩展；LightGBM 每次选择全树中增益最大的叶子继续分裂。它常能更快降低训练损失，但若 `num_leaves`、`max_depth` 和 `min_child_samples` 约束不足，也更易过拟合。

### 2.3 GOSS 与 EFB

GOSS 优先保留大梯度样本，并抽取部分小梯度样本；EFB 将很少同时非零的稀疏特征打包。二者是效率技术，并非每种配置都必须启用。

## 3. 数学形式

### 3.1 加法模型

$$
F_m(x)=F_{m-1}(x)+\nu f_m(x)
$$

### 3.2 直方图统计量

若样本被分入 bin $b$，则：

$$
G_b=\sum_{i\in b}g_i,\qquad
H_b=\sum_{i\in b}h_i
$$

扫描 bin 的前缀和即可得到候选左右节点的 $G_L,H_L,G_R,H_R$。

### 3.3 分裂增益

一个常见的正则化增益形式为：

$$
\operatorname{Gain}=
\frac12\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma
$$

具体实现还会应用叶节点最小样本数、最小 Hessian、最大深度和最小增益等约束。

### 3.4 关键参数

- `num_leaves`：单树最大叶数，核心复杂度参数。
- `max_depth`、`min_child_samples`：限制 leaf-wise 过度生长。
- `max_bin`：直方图分箱数量。
- `feature_fraction`、`bagging_fraction`：列、行采样。
- `learning_rate`、`num_iterations`：步长与轮数。

## 4. 手把手算例

某连续特征被分成 3 个 bin，每个 bin 含 2 人。6 人的一阶梯度为：

$$
g=(-0.6,-0.5,-0.1,0.1,0.2,0.3)
$$

每人的 Hessian 均为 $0.25$，取 $\lambda=1,\gamma=0$。

**Step 1：汇总 bins。**

| bin | 梯度和 $G_b$ | Hessian 和 $H_b$ |
| --- | ---: | ---: |
| 1 | -1.1 | 0.5 |
| 2 | 0.0 | 0.5 |
| 3 | 0.5 | 0.5 |

父节点 $G=-0.6,H=1.5$。

**Step 2：尝试在 bin 1 后分裂。**

左侧 $G_L=-1.1,H_L=0.5$；右侧 $G_R=0.5,H_R=1.0$：

$$
\operatorname{Gain}_1=
\frac12\left(
\frac{1.21}{1.5}+\frac{0.25}{2.0}-\frac{0.36}{2.5}
\right)\approx0.394
$$

**Step 3：尝试在 bin 2 后分裂。**

左侧 $G_L=-1.1,H_L=1.0$；右侧 $G_R=0.5,H_R=0.5$：

$$
\operatorname{Gain}_2=
\frac12\left(
\frac{1.21}{2.0}+\frac{0.25}{1.5}-\frac{0.36}{2.5}
\right)\approx0.314
$$

因此选择 bin 1 后的阈值。随后 leaf-wise 策略会比较当前所有叶子的下一次候选增益，只分裂增益最大的那个叶子。

**结论：** LightGBM 用 bin 汇总量近似完成阈值搜索，并用全局最大叶增益决定生长位置。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 适用于分类、回归和排序等表格任务。
- 可接受数值、稀疏输入及按接口声明的类别特征。
- 可原生处理缺失，但医学研究仍需分析缺失机制和流程信息泄漏。
- 重复测量必须按患者或中心合理分组切分。

### 5.2 产出

输出概率、类别、连续预测或排序分数，以及最佳轮数、训练记录和特征重要性。它不直接提供传统回归参数推断。

## 6. 适用场景

- 大样本、高维稀疏电子病历或多模态特征表。
- 训练效率和频繁调参很重要的预测任务。
- 慎用于小样本、外推、因果解释或未经控制的数据泄漏场景。

## 7. 实现

### 7.1 Python

```python
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, brier_score_loss

model = LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="binary_logloss",
    callbacks=[early_stopping(50), log_evaluation(0)],
)

prob = model.predict_proba(X_test)[:, 1]
print(model.best_iteration_)
print(roc_auc_score(y_test, prob))
print(brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(lightgbm)

dtrain <- lgb.Dataset(as.matrix(X_train), label = y_train)
dvalid <- lgb.Dataset(as.matrix(X_valid), label = y_valid)

fit <- lgb.train(
  params = list(
    objective = "binary",
    metric = "binary_logloss",
    learning_rate = 0.03,
    num_leaves = 31,
    min_data_in_leaf = 30,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 1
  ),
  data = dtrain,
  nrounds = 3000,
  valids = list(valid = dvalid),
  early_stopping_rounds = 50,
  verbose = -1
)

prob <- predict(fit, as.matrix(X_test))
```

## 8. 结果如何解释

- `best_iteration_` 是验证集早停轮数。
- split importance 是分裂次数，gain importance 是累计增益，两者含义不同。
- `num_leaves` 越大不是越好，应结合叶节点最小样本数与样本量解释。
- 模型解释图反映预测依赖，不应写成可干预因果效应。

## 9. 诊断与稳健性

1. 检查训练/验证曲线和最佳轮数。
2. 联合调节 `num_leaves`、`min_child_samples`、`max_depth`。
3. 比较不同 `max_bin`、随机种子和数据切分的稳定性。
4. 核查类别特征编码、缺失与时间泄漏。
5. 进行中心外、时间外、关键亚组、校准和决策曲线评估。

## 10. 推荐可视化

- 训练/验证损失曲线。
- ROC、PR、校准和决策曲线。
- gain 与 permutation importance 对照图。
- SHAP 总结图和局部解释图。

下图给出前 20 个特征的重要性排序：

![](../../04_示例图像/lightgbm_feature_importance_top20.png)

## 11. 优势、局限与常见坑

**优势：** 训练快、内存效率高、适合高维稀疏表格数据，并支持类别和缺失特征的高效处理。

**局限：** leaf-wise 易在小样本中过拟合，超参数较多，概率和解释仍需独立验证。

**常见坑：** `num_leaves` 过大；设置 `subsample` 却未启用采样频率；用测试集早停；把训练速度等同于模型质量。

## 12. 与相近方法的区别

- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]：两者都是现代提升树，LightGBM 更突出 histogram 与 leaf-wise 效率。
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]：LightGBM 是高效工程化实现，不是不同的基本学习范式。
- [[随机森林（Random Forest）]]：随机森林并行平均，LightGBM 串行纠错。
- [[极端随机树（Extremely Randomized Trees）]]：后者随机阈值并聚合独立树，不拟合负梯度。

## 13. 医学研究中的典型应用

- 大规模电子病历恶化、死亡与再入院风险预测。
- 多中心临床数据和高维稀疏诊断编码建模。
- 临床、组学、影像派生特征融合及快速模型迭代。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| histogram | 将连续取值离散到有限 bins 后汇总梯度 |
| leaf-wise | 每次优先分裂全树中增益最大的叶子 |
| GOSS | 偏重保留大梯度样本的采样策略 |
| EFB | 打包互斥稀疏特征以减少有效维度 |
| `num_leaves` | 单棵树允许的最大叶节点数 |

## 15. 相关方法

- [[Boosting算法（Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[随机森林（Random Forest）]]

## 16. 参考资料

- Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. In: *Advances in Neural Information Processing Systems 30*. 2017.
- LightGBM Contributors. LightGBM Documentation. [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/) （访问日期：2026-07-09）
- Shi X, Wong YD, Li MZ, Palanisamy C, Chai C. A feature learning approach based on XGBoost for driving assessment and risk prediction. *Accid Anal Prev*. 2019;129:170-179.

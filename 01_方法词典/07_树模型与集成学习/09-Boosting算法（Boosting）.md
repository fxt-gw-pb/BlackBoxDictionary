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

### 1.1 一句话本质

Boosting 让弱学习器依次上场，每一轮都针对当前模型尚未解决的错误做一次小修正，最后把这些修正相加成强模型。

### 1.2 定义

Boosting 是一类串行集成学习框架。第一个弱学习器给出粗略预测，后续学习器根据错分样本、残差或损失函数负梯度继续学习，再按权重或学习率叠加。[[AdaBoost（Adaptive Boosting）]]、[[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]、[[XGBoost（Extreme Gradient Boosting, XGBoost）]] 和 [[LightGBM（Light Gradient Boosting Machine）]] 都属于该家族。

### 1.3 它主要解决什么问题

- 如何把多个仅略有预测能力的简单模型组合为高性能模型。
- 如何逐步降低单个浅树难以解决的偏差。
- 如何学习表格数据中的非线性、阈值效应与交互。

### 1.4 直觉与类比

它像连续批改同一份答卷：第一位老师标出主要问题，第二位专看第一位遗漏的错误，后面的人继续补漏。最终答案不是多数投票，而是“初稿加上一连串修订”。

## 2. 核心思想与原理

### 2.1 根本困难

浅树稳定、易控制，却往往欠拟合；深树能拟合复杂模式，却容易高方差。Boosting 采用多棵浅树，每次只走一小步，在降低偏差的同时用学习率、树深和早停限制复杂度。

### 2.2 三种“关注错误”的方式

- AdaBoost：提高错分样本的权重。
- 梯度提升：拟合当前损失的负梯度；平方损失下就是残差。
- 现代提升树：使用一阶或二阶梯度、正则化和采样策略选择分裂。

### 2.3 串行性为何重要

第 $m$ 个学习器依赖前 $m-1$ 个学习器的预测，因此不能像 [[Bagging算法（Bootstrap Aggregating）]] 那样完全独立训练。它通常更擅长降偏差，也更需要验证集监控。

## 3. 数学形式

### 3.1 加法模型

$$
F_M(x)=F_0(x)+\sum_{m=1}^{M}\nu\alpha_m h_m(x)
$$

其中 $h_m$ 是第 $m$ 个弱学习器，$\alpha_m$ 是其权重或线搜索步长，$\nu$ 是学习率。

### 3.2 逐轮优化

一般形式是在每轮寻找一个弱学习器和步长，使经验损失下降：

$$
(\alpha_m,h_m)=
\operatorname*{arg\,min}_{\alpha,h}
\sum_{i=1}^{n}
L\left[y_i,F_{m-1}(x_i)+\alpha h(x_i)\right]
$$

再更新：

$$
F_m(x)=F_{m-1}(x)+\nu\alpha_mh_m(x)
$$

### 3.3 关键参数

- `n_estimators`：最大提升轮数。
- `learning_rate`：每轮修正幅度；越小通常需要更多轮。
- 基学习器复杂度：浅树控制单轮自由度。
- 损失函数：决定“错误”如何被度量。
- 子采样与早停：用于抑制过拟合。

### 3.4 关键假设(含违反后果)

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 弱学习器能捕捉剩余结构 | 多轮后性能仍不改善 | 画验证损失曲线 |
| 标签与测量误差可控 | 模型反复追逐噪声 | 审核高权重或大残差样本 |
| 训练人群代表目标人群 | 外部性能明显下降 | 时间或外部验证 |
| 损失与临床目标一致 | 指标提高但临床效用不佳 | 同时评估校准与决策曲线 |

## 4. 手把手算例

预测 4 名患者的连续结局，真实值为 $y=(2,4,6,8)$。

**Step 1：初始化。** 平方损失下，最佳常数是均值：

$$
F_0=\bar y=5
$$

初始残差为 $(-3,-1,1,3)$，均方误差为：

$$
\operatorname{MSE}_0=\frac{9+1+1+9}{4}=5
$$

**Step 2：第一棵浅树。** 假设树把前两人与后两人分组，叶节点分别拟合残差均值 $-2$ 与 $2$。取 $\nu=0.5$：

$$
F_1=(5,5,5,5)+0.5(-2,-2,2,2)=(4,4,6,6)
$$

此时残差为 $(-2,0,0,2)$，$\operatorname{MSE}_1=2$。

**Step 3：第二棵树。** 假设新树拟合出 $h_2=(-2,0,0,2)$：

$$
F_2=F_1+0.5h_2=(3,4,6,7)
$$

于是 $\operatorname{MSE}_2=(1+0+0+1)/4=0.5$。

**结论：** 每棵树都不必一次给出最终答案；只要持续拟合尚未解释的部分，集成误差就能逐轮下降。

## 5. 数据形式与输入输出

### 5.1 适合的数据

- 输入：数值特征、编码后的类别特征和目标变量。
- 结局：二分类、多分类或连续型，取决于具体算法与损失。
- 缺失：经典实现常需插补；XGBoost、LightGBM 可学习缺失默认方向。
- 不直接处理：聚类结局、删失与重复测量相关性，除非使用专门扩展。

### 5.2 输入与产出

输入包括训练集、损失函数、弱学习器、轮数和学习率。产出包括预测值或概率、逐轮训练记录和变量重要性。传统标准误、置信区间和可直接解释的回归系数通常不是默认产物。

## 6. 适用场景

- 适合：非线性与交互明显、预测性能优先的表格数据。
- 慎用：极小样本、标签噪声高、外推范围大或主要目标是因果效应估计。
- 医学研究中应配合独立测试集、概率校准、临床阈值和外部验证。

## 7. 实现

### 7.1 Python

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=2,
    n_iter_no_change=20,
    validation_fraction=0.2,
    random_state=42,
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
print("trees:", model.n_estimators_)
print("AUC:", roc_auc_score(y_test, prob))
print("Brier:", brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(gbm)

fit <- gbm(
  Sepsis ~ Age + HR + SBP + Lactate + WBC,
  data = train,
  distribution = "bernoulli",
  n.trees = 500,
  interaction.depth = 2,
  shrinkage = 0.03,
  bag.fraction = 0.8,
  cv.folds = 5
)

best_n <- gbm.perf(fit, method = "cv", plot.it = FALSE)
prob <- predict(fit, newdata = test, n.trees = best_n, type = "response")
```

## 8. 结果如何解释

- 首先报告独立验证集的判别、误差或校准，而不是训练损失。
- 学习率与树数应成对解释：小学习率加较多树通常更平滑。
- 特征重要性表示模型依赖程度，不是独立效应或因果效应。
- 分类概率若用于临床决策，应检查校准截距、校准斜率和阈值净获益。

## 9. 诊断与稳健性

1. 绘制训练和验证损失随轮数变化，确定早停点。
2. 比较不同随机种子、树深和学习率下性能稳定性。
3. 审查持续大残差或被反复错分的病例，区分困难样本与错误标签。
4. 进行时间切分、中心外验证和亚组评估。
5. 对概率模型检查校准曲线、Brier 分数和决策曲线。

## 10. 推荐可视化

- 训练/验证损失曲线。
- ROC、PR 与校准曲线。
- 预测值-真实值图和残差图。
- permutation importance、部分依赖或 SHAP 图；解释时注明其非因果性质。

## 11. 优势、局限与常见坑

**优势：** 表格数据性能强，能学习非线性与交互，损失函数灵活。

**局限：** 串行训练、调参维度多、对噪声与分布漂移敏感，可解释性弱于参数模型。

**常见坑：** 学习率过大、树过深；用测试集早停；只报 AUC；把训练误差持续下降误认为泛化持续改善。

## 12. 与相近方法的区别

- [[Bagging算法（Bootstrap Aggregating）]]：并行拟合后平均，主要降方差。
- [[随机森林（Random Forest）]]：依靠 bootstrap 和随机特征去相关。
- [[AdaBoost（Adaptive Boosting）]]：以错分样本重加权为代表。
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]：以损失负梯度统一每轮修正。

## 13. 医学研究中的典型应用

- 败血症、死亡、再入院和术后并发症风险预测。
- 住院时长、费用和连续生理指标预测。
- 影像组学、组学与临床变量融合后的表格建模。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| 弱学习器 | 单独表现有限、通常受复杂度约束的模型 |
| 加法模型 | 预测函数由多个学习器加权相加 |
| 学习率 | 缩小每轮更新幅度的系数 |
| 伪残差 | 损失对当前预测的负梯度 |
| 早停 | 验证性能不再改善时停止迭代 |

## 15. 相关方法

- [[AdaBoost（Adaptive Boosting）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 16. 参考资料

- Schapire RE. The strength of weak learnability. *Mach Learn*. 1990;5:197-227.
- Freund Y, Schapire RE. A decision-theoretic generalization of on-line learning and an application to boosting. *J Comput Syst Sci*. 1997;55(1):119-139.
- Friedman JH. Greedy function approximation: a gradient boosting machine. *Ann Stat*. 2001;29(5):1189-1232.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.

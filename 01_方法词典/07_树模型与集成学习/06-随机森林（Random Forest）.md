---
title: 随机森林
english_name: Random Forest
slug: random-forest
aliases: [random forest, RF, "随机森林（Random Forest）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 树模型, 分类]
status: 已建
difficulty: basic
question_type: 非线性分类与回归预测
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ranger, randomForest]
---

# 随机森林（Random Forest）

## 1. 方法概览

### 1.1 一句话本质

随机森林是在 Bagging 多棵树的基础上，再让每个节点只看一部分随机特征，从而让树与树更不相似、投票更稳。

### 1.2 定义

随机森林是一种基于决策树的集成学习方法。它对样本做 bootstrap 抽样，对节点分裂候选特征做随机子采样，训练许多不同的树，再通过投票或概率平均给出分类预测。

### 1.3 它主要解决什么问题

- 研究问题：如何在非线性和交互复杂的表格数据中，获得比单棵树更稳定的分类模型。
- 适用任务：二分类、多分类、风险预测、变量重要性初探。
- 常见医学场景：再入院风险、死亡风险、并发症风险、疾病分型辅助分类。

### 1.4 直觉与类比

随机森林像组建一个专家委员会，但不让每位专家都看完全相同的病例和指标。每位专家都见过不同的 bootstrap 病例，而且每次分裂时只允许看一部分指标。这样每棵树会形成不同视角，集体投票更不容易被某个偶然模式带偏。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

Bagging 已经能平均多棵树，但如果所有树都被同一个强预测变量支配，它们仍会长得很像，错误也高度相关。根本困难是：**仅靠样本重抽样还不一定能让树足够多样，相关树的平均降方差效果有限**。

### 2.2 关键洞察

随机森林的关键洞察是给树再加一层「特征随机性」。每个节点分裂时，只在随机抽取的一小部分特征中寻找最佳分裂，迫使不同树探索不同变量组合。树与树相关性降低后，投票或平均更能降低方差。

### 2.3 与朴素/相邻做法的对比

- 相对 [[决策树（Decision Tree）]]：随机森林用多棵树投票，显著降低单树不稳定性。
- 相对 [[Bagging算法（Bootstrap Aggregating）]]：随机森林不仅抽样本，还在节点抽特征，进一步降低树间相关性。
- 相对 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]：随机森林并行建树，主要降方差；GBDT 串行加树，逐步修正残差或梯度。

## 3. 数学形式

### 3.1 核心公式

设共有 $B$ 棵树 $T_b(x)$。分类预测为：

$$
\hat y=
\operatorname*{arg\,max}_{k}
\sum_{b=1}^{B}I\{T_b(x)=k\}
$$

类别概率可用投票比例估计：

$$
\hat p_k(x)=\frac{1}{B}\sum_{b=1}^{B}I\{T_b(x)=k\}
$$

每个节点分裂时，若总特征数为 $p$，只从 $m_{\mathrm{try}}$ 个随机特征中选择最佳分裂：

$$
m_{\mathrm{try}}\approx \sqrt{p}\quad \text{(分类常用经验)}
$$

这个式子在说：预测来自多树投票，而每棵树在成长时都被迫只看部分特征。

### 3.2 推导脉络

1. 对训练集做 bootstrap 抽样，生成第 $b$ 棵树的训练样本。
2. 在该树的每个节点，随机抽取 $m_{\mathrm{try}}$ 个候选特征。
3. 只在这批候选特征中寻找最优分裂。
4. 树通常长得较深，不强剪枝。
5. 对所有树的类别预测投票，得到最终类别和概率。

### 3.3 参数与统计量含义

- `n_estimators`：树的数量。
- `max_features` / $m_{\mathrm{try}}$：每个节点候选特征数，控制树间相关性。
- `min_samples_leaf`：叶节点最小样本数，影响概率平滑程度。
- OOB 误差：用袋外样本估计泛化误差。
- 特征重要性：衡量变量对分裂纯度或预测性能的贡献，但不是因果效应。

### 3.4 关键假设(含违反后果)

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 树能捕捉信号 | 非线性和交互可由分裂表达 | 森林也学不到有效模式 | 外部验证、学习曲线 |
| 树之间足够多样 | 样本和特征随机降低相关性 | 多树收益有限 | 调整 `max_features` |
| 训练数据代表目标人群 | 森林只降低采样波动 | 人群漂移下性能下降 | 时间外/中心外验证 |
| 类别不平衡被处理 | 少数类有足够信号 | 默认投票偏向多数类 | PR-AUC、召回率、类权重 |
| 概率需校准 | 投票比例不一定是真概率 | 风险阈值决策失准 | 校准曲线、Brier score |

## 4. 手把手算例

### 4.1 多树投票

3 棵随机森林中的树对 4 位患者 30 天再入院作出预测：

| 患者 | 树1 | 树2 | 树3 | 预测概率 $\hat p(Y=1)$ | 最终类别 |
| --- | --- | --- | --- | --- | --- |
| A | 1 | 1 | 0 | 2/3=0.67 | 1 |
| B | 0 | 0 | 1 | 1/3=0.33 | 0 |
| C | 1 | 1 | 1 | 3/3=1.00 | 1 |
| D | 0 | 1 | 0 | 1/3=0.33 | 0 |

多数投票给出类别，投 1 的比例可作为未经校准的风险概率。

### 4.2 为什么要随机特征

假设每棵树方差 $\sigma^2=9$、树数 $B=100$。若普通 Bagging 中树相关性 $\rho=0.4$：

$$
9\left(0.4+\frac{0.6}{100}\right)=3.654
$$

随机森林通过特征随机使相关性降到 $\rho=0.15$：

$$
9\left(0.15+\frac{0.85}{100}\right)=1.427
$$

**结论：** 随机森林的关键不只是「树多」，而是「树之间别太像」。特征随机降低相关性，集成方差就进一步下降。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可。
- 因变量类型：二分类或多分类；框架也可用于连续结局。
- 数据结构：宽表数据。
- 是否适合高维数据：可用，但极高维稀疏问题未必最优。
- 是否适合缺失较多数据：通常建议先系统处理缺失。
- 是否适合删失数据：原始随机森林不适合，需使用生存森林变体。
- 是否适合重复测量数据：不直接适合。

### 5.2 示例表格

| Age | Charlson | HbA1c | eGFR | PriorAdmission | Readmit30d |
| --- | --- | --- | --- | --- | --- |
| 71 | 4 | 8.9 | 48 | 1 | 1 |
| 52 | 1 | 6.7 | 92 | 0 | 0 |
| 63 | 3 | 7.8 | 55 | 1 | 1 |
| 44 | 0 | 6.1 | 101 | 0 | 0 |
| 68 | 2 | 7.1 | 60 | 0 | 0 |

### 5.3 输入与产出

#### 输入

- 输入数据：目标变量和特征矩阵。
- 关键变量：树数、最大深度、最小叶节点、最大特征数、类权重。
- 需要预处理的内容：缺失处理、训练测试集划分、类别不平衡处理。

#### 产出

- 模型对象/统计结果：森林、OOB 误差、特征重要性。
- 参数估计：不提供传统回归系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：OOB 误差、测试集 AUC/F1、PR-AUC、校准指标。

## 6. 适用场景

- 适合：表格数据、非线性和交互复杂、以预测性能为导向的任务。
- 不适合：强解释导向、样本极小、需要稳定可解释概率或简单规则的场景。
- 使用前需要特别检查的点：类别不平衡、变量重要性偏倚、概率校准、外部验证。

## 7. 实现

### 7.1 Python

常用包:

- `scikit-learn`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
print(model.oob_score_)
print(roc_auc_score(y_test, prob))
```

### 7.2 R

常用包:

- `ranger`

```r
library(ranger)

fit <- ranger(
  Readmit30d ~ Age + Charlson + HbA1c + eGFR + PriorAdmission,
  data = train_df,
  probability = TRUE,
  num.trees = 500,
  mtry = 2,
  min.node.size = 10,
  importance = "permutation"
)

pred <- predict(fit, data = test_df)$predictions[, "1"]
```

## 8. 结果如何解读

- 核心结果看什么：测试集/外部验证性能、OOB 误差、校准、变量重要性。
- 每个主要参数如何解读：树数增多通常更稳定；`max_features` 越小，树之间越不同但单树可能更弱。
- 临床或医学意义如何表达：可说「模型主要依赖这些变量进行风险预测」，不要说「这些变量导致结局」。
- 常见误读：特征重要性高不等于因果作用；OOB 好不等于外部数据一定好。

## 9. 假设诊断与稳健性

- OOB 与测试集对照：OOB 可作内部估计，最终仍看独立测试或外部验证。
- 校准诊断：用校准曲线和 Brier score 检查预测概率。
- 类别不平衡：报告敏感度、特异度、PR-AUC，必要时调类权重和阈值。
- 变量重要性稳健性：优先看 permutation importance，并用重采样看排序是否稳定。
- 超参数敏感性：检查 `max_features`、`min_samples_leaf` 和树数对性能的影响。

## 10. 推荐可视化

- 特征重要性条形图。
- ROC 曲线、PR 曲线和校准曲线。
- OOB 误差随树数变化曲线。
- 部分依赖图或 SHAP 图，用于辅助解释预测模式。

### 10.1 图像示例

下图展示随机森林分类案例中的特征重要性排序，用来观察模型主要依赖哪些输入变量完成判别。

![](../../04_示例图像/random_forest_classifier_feature_importance.png)

## 11. 优势、局限与常见坑

### 优势

- 对非线性和交互建模能力强。
- 通常比单棵树稳定，调参相对稳健。
- 对特征缩放不敏感。

### 局限

- 可解释性弱于单棵树和参数模型。
- 变量重要性可能偏向连续或高基数变量。
- 预测概率常需要额外校准。

### 常见坑

- 只看内部 OOB 表现，不做外部验证。
- 把变量重要性当作因果证据。
- 在强类别不平衡下仍沿用默认 0.5 阈值。
- 忽略概率校准，直接用于临床风险阈值。

## 12. 与相近方法的区别

- 和 [[决策树（Decision Tree）]] 的区别：随机森林通过多树集成降方差，通常泛化更好。
- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：随机森林在 Bagging 基础上增加节点级特征随机。
- 和 [[随机森林回归（Random Forest Regression）]] 的区别：本卡聚焦分类投票和概率；回归版本聚焦连续值平均。
- 和 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]] 的区别：随机森林是并行 bagging，GBDT 是串行 boosting。
- 如何选择：想要稳健表格分类基线时用随机森林；若追求更高性能可比较 GBDT/XGBoost。

## 13. 医学研究中的典型应用

- 再入院、死亡、并发症等临床风险预测。
- 生物标志物筛选与变量重要性初探。
- 多特征疾病分类与辅助决策。

## 14. 关键术语

- **随机森林（Random Forest）**：由多棵随机化决策树组成的集成模型。
- **特征随机子采样（Feature Subsampling）**：每个节点只从部分特征中选分裂。
- **袋外误差（Out-of-Bag Error）**：用未进入某棵树 bootstrap 样本的病例估计误差。
- **变量重要性（Variable Importance）**：衡量变量对预测或分裂贡献的指标。
- **Permutation Importance**：打乱某变量后性能下降的幅度。
- **类权重（Class Weight）**：给少数类更大损失权重以处理类别不平衡。
- **校准曲线（Calibration Curve）**：比较预测概率与实际发生率的图。

## 15. 相关方法

- [[Bagging算法（Bootstrap Aggregating）]]
- [[决策树（Decision Tree）]]
- [[随机森林回归（Random Forest Regression）]]
- [[极端随机树（Extremely Randomized Trees）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]

## 16. 参考资料

- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- Breiman L. Bagging predictors. *Mach Learn*. 1996;24:123-140.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- Wright MN, Ziegler A. ranger: A fast implementation of random forests for high dimensional data in C++ and R. *J Stat Softw*. 2017;77(1):1-17.

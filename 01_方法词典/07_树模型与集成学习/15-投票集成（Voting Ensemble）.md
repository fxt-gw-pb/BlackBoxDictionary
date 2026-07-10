---
title: 投票集成
english_name: Voting Ensemble
slug: voting-ensemble
aliases: [voting ensemble, voting classifier, 投票法, "投票集成（Voting Ensemble）"]
category: 树模型与集成学习
subcategory: 模型融合
tags: [医学统计, 数据科学, 集成学习, 模型融合, 分类]
status: 已建
difficulty: basic
question_type: 多模型分类融合
data_type: [表格数据]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn]
r_packages: [caret, caretEnsemble]
---

# 投票集成（Voting Ensemble）

## 1. 方法概览

### 1.1 一句话本质

投票集成用固定规则合并多个分类模型：硬投票数类别票，软投票平均经过校准且可比较的类别概率。

### 1.2 定义

投票集成是一种并行模型融合方法。各基模型独立训练，对同一患者产生类别或概率，再用多数票、平均概率或预先确定的权重得到最终结果。

### 1.3 它主要解决什么问题

- 多个模型性能接近，却在不同患者上犯不同错误。
- 希望用低复杂度规则提高预测稳定性。
- 需要融合临床、影像或组学等不同建模管线的输出。

### 1.4 直觉与类比

像一个诊断会诊组：硬投票只看每位专家的最终结论；软投票还看每位专家给出的风险概率。若专家们错误互补，会诊可能更稳；若所有人依据相同信息犯同样错误，人多也无济于事。

## 2. 核心思想与原理

### 2.1 集成收益来自哪里

收益来自**准确性与多样性的共同作用**。基模型要有基本能力，预测误差又不能完全相关。加入一个与已有模型高度相似的模型，通常只是在重复计票。

### 2.2 硬投票与软投票

- 硬投票：只使用类别，简单但丢掉置信程度。
- 软投票：使用概率，信息更多，但要求各模型概率含义与校准程度可比。
- 加权投票：权重必须在训练数据内部通过交叉验证确定，不能看测试集后再调。

### 2.3 为什么校准重要

若一个过度自信的模型经常输出 0.99，另一个校准良好的模型只输出 0.70，直接平均会让前者获得不成比例的影响。软投票前应比较校准曲线、Brier 分数，必要时在训练流程内校准。

## 3. 数学形式

### 3.1 硬投票

$$
\hat y=
\operatorname*{arg\,max}_{c\in\mathcal C}
\sum_{m=1}^{M}w_mI\{h_m(x)=c\}
$$

### 3.2 软投票

$$
\bar p(c\mid x)=
\frac{\sum_{m=1}^{M}w_mp_m(c\mid x)}
{\sum_{m=1}^{M}w_m}
$$

$$
\hat y=
\operatorname*{arg\,max}_{c\in\mathcal C}\bar p(c\mid x)
$$

### 3.3 参数含义

- $M$：基模型数。
- $w_m$：第 $m$ 个模型的非负权重。
- $p_m(c\mid x)$：模型对类别 $c$ 的概率。
- 阈值：二分类时不必固定为 0.5，应由临床成本与验证数据决定。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 基模型有基本预测能力 | 弱模型拖累集成 | 与最佳单模型比较 |
| 错误具有互补性 | 集成几乎无收益 | 预测相关性与分歧率 |
| 软投票概率可比较 | 过度自信模型支配结果 | 校准曲线、Brier 分数 |
| 权重不使用测试集调节 | 测试性能乐观偏倚 | 固定嵌套验证流程 |

## 4. 手把手算例

三模型对某患者的阳性概率分别为：

| 模型 | 阳性概率 | 以 0.5 分类 |
| --- | ---: | ---: |
| Logistic 回归 | 0.40 | 0 |
| 随机森林 | 0.55 | 1 |
| SVM | 0.90 | 1 |

**Step 1：硬投票。** 两票阳性、一票阴性，所以最终类别为阳性。

**Step 2：等权软投票。**

$$
\bar p=\frac{0.40+0.55+0.90}{3}=0.617
$$

仍判为阳性。

**Step 3：加权软投票。** 若训练内验证后给 Logistic 回归权重 2，其余各 1：

$$
\bar p_w=
\frac{2(0.40)+0.55+0.90}{2+1+1}
=0.563
$$

结果仍为阳性，但风险降低。

**Step 4：看校准风险。** 若 SVM 的 0.90 来自明显过度自信的未校准分数，0.617 并非可靠临床概率。先校准再平均，才有概率解释。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 二分类或多分类监督学习。
- 基模型可使用相同或不同特征，但预测必须对应同一批患者与同一结局定义。
- 所有预处理、特征选择和校准都必须包含在训练管线内。

### 5.2 输入与产出

输入为基模型、投票方式和预先定义的权重。输出为最终类别；软投票还输出平均概率。它不产生传统回归系数，也不自动给出预测不确定性。

## 6. 适用场景

- 已有多个性能合格且错误互补的分类模型。
- 需要比 Stacking 更简单、可审计的融合规则。
- 不适合所有基模型都弱、都来自几乎相同算法或概率严重失准的场景。

## 7. 实现

### 7.1 Python

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, brier_score_loss

models = [
    ("lr", make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )),
    ("rf", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=42
    )),
    ("svm", make_pipeline(
        StandardScaler(),
        SVC(C=1, probability=True, random_state=42)
    )),
]

ensemble = VotingClassifier(
    estimators=models,
    voting="soft",
    weights=[2, 1, 1],
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)
prob = ensemble.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob))
print(brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(caret)
library(caretEnsemble)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  savePredictions = "final",
  summaryFunction = twoClassSummary
)

models <- caretList(
  Malignant ~ .,
  data = train,
  trControl = ctrl,
  metric = "ROC",
  methodList = c("glm", "rf")
)

ensemble <- caretEnsemble(
  models,
  metric = "ROC",
  trControl = ctrl
)
prob <- predict(ensemble, newdata = test)
```

## 8. 结果如何解释

- 先比较集成与最佳单模型的独立测试表现。
- 软投票概率是加权平均，只有输入概率可信时才可作临床风险解释。
- 权重表示模型在固定融合规则中的影响，不是原始变量的重要性。
- 报告最终阈值、敏感度、特异度、PPV、NPV、校准和净获益。

## 9. 诊断与稳健性

1. 比较各模型与集成的 AUC、PR-AUC、Brier 和校准。
2. 计算模型概率相关矩阵与病例级分歧。
3. 做 leave-one-model-out 分析，观察移除某模型的影响。
4. 用嵌套交叉验证选择权重，并只在最终测试集评估一次。
5. 检查时间外、中心外和关键亚组表现。

## 10. 推荐可视化

- 单模型与集成的 ROC、PR 和校准曲线。
- 基模型预测概率相关性热图。
- 病例级预测分歧图。
- 权重与 leave-one-model-out 性能图。

## 11. 优势、局限与常见坑

**优势：** 简单、并行、易复现，可融合异质模型。

**局限：** 不学习复杂组合关系，收益依赖多样性，软投票依赖校准。

**常见坑：** 用测试集调权重；把决策分数当概率；让多个近乎相同的模型重复计票；只报告集成而不报告单模型。

## 12. 与相近方法的区别

- [[Stacking集成（Stacked Generalization）]]：用元学习器学习组合规则，更灵活也更易过拟合。
- [[Bagging算法（Bootstrap Aggregating）]]：通常聚合同类模型的重抽样版本。
- [[Boosting算法（Boosting）]]：串行训练后续模型以纠正前序错误。
- 选择经验：数据量有限、希望规则透明时先用投票；有足够样本和严格 OOF 流程时再考虑 Stacking。

## 13. 医学研究中的典型应用

- 融合临床、实验室与影像组学分类模型。
- 汇总不同算法对肺结节、肿瘤亚型或并发症的判断。
- 多中心模型的稳健预测融合。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| hard voting | 对预测类别计票 |
| soft voting | 对类别概率求平均 |
| calibration | 预测概率与真实发生率的一致程度 |
| model diversity | 不同模型错误模式的差异程度 |
| weighted voting | 按预定权重合并预测 |

## 15. 相关方法

- [[Stacking集成（Stacked Generalization）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[Boosting算法（Boosting）]]
- [[随机森林（Random Forest）]]

## 16. 参考资料

- Dietterich TG. Ensemble methods in machine learning. In: *Multiple Classifier Systems*. 2000:1-15.
- Zhou ZH. *Ensemble Methods: Foundations and Algorithms*. Chapman and Hall/CRC; 2012.
- scikit-learn Developers. `VotingClassifier` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) （访问日期：2026-07-09）

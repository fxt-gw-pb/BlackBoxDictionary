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

### 1.1 定义

投票集成是一种把多个分类模型的预测结果合并为最终分类结果的模型融合方法。硬投票统计类别票数，软投票累加或加权平均类别概率。

### 1.2 它主要解决什么问题

- 研究问题：当多个分类模型各有优势时，如何得到更稳健的最终判断。
- 适用任务：二分类、多分类预测。
- 常见医学场景：多模型疾病诊断、风险分层、不同特征组模型的融合。

### 1.3 直觉理解

投票集成像让多个模型组成评审团。硬投票看“多数人选什么”，软投票看“综合置信度最高的是哪一类”。当模型错误模式互补时，集成结果往往比单个模型更稳。

## 2. 数学形式

### 2.1 核心公式

硬投票中，第 $m$ 个模型的分类结果为 $h_m(x)$，类别集合为 $\mathcal{C}$，最终预测为：

$$
\hat y=\arg\max_{c\in\mathcal{C}}\sum_{m=1}^{M}I(h_m(x)=c)
$$

软投票中，第 $m$ 个模型输出类别概率 $p_m(c\mid x)$，最终预测为：

$$
\hat y=\arg\max_{c\in\mathcal{C}}\sum_{m=1}^{M}p_m(c\mid x)
$$

若使用权重 $w_m$，则：

$$
\hat y=\arg\max_{c\in\mathcal{C}}\sum_{m=1}^{M}w_m p_m(c\mid x)
$$

### 2.2 参数或统计量含义

- $M$：参与投票的基模型数量。
- $p_m(c\mid x)$：第 $m$ 个模型给出的类别概率。
- $w_m$：模型权重，常由交叉验证性能确定。
- 硬投票：只使用类别标签。
- 软投票：使用概率输出，依赖概率校准质量。

### 2.3 关键假设

- 基模型至少有一定预测能力。
- 基模型之间存在互补信息，错误不完全相关。
- 软投票要求概率输出具有可比性，最好经过校准。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：取决于基模型，可包含表格、文本向量、影像特征等。
- 因变量类型：二分类或多分类。
- 数据结构：监督学习数据。
- 是否适合高维数据：取决于基模型组合。
- 是否适合缺失较多数据：需在各基模型前统一处理。
- 是否适合删失数据：普通投票分类不直接适合。
- 是否适合重复测量数据：需先构造个体级预测或使用适配基模型。

### 3.2 示例表格

以肺结节良恶性分类为例：

| Age | NoduleSize | Spiculation | Smoking | RadiomicsScore | Malignant |
| --- | --- | --- | --- | --- | --- |
| 67 | 18 | 1 | 1 | 0.72 | 1 |
| 45 | 6 | 0 | 0 | 0.18 | 0 |
| 59 | 12 | 1 | 1 | 0.54 | 1 |
| 52 | 8 | 0 | 1 | 0.31 | 0 |
| 73 | 22 | 1 | 1 | 0.81 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：同一批样本的特征矩阵和分类标签。
- 关键变量：基模型列表、投票方式、模型权重、交叉验证方案。
- 需要预处理的内容：缺失处理、编码、缩放、概率校准、训练测试划分。

#### 产出

- 模型对象/统计结果：投票集成分类器、各基模型表现、权重。
- 参数估计：通常不输出传统系数。
- 预测结果：最终类别和概率。
- 不确定性指标：交叉验证性能、测试集 AUC / F1、校准曲线。

## 4. 适用场景

- 适合：已有多个互补模型，且希望快速融合分类结果的场景。
- 不适合：所有基模型表现都很弱或错误高度相关的场景。
- 使用前需要特别检查的点：软投票概率是否校准、权重是否仅在训练内部确定、是否发生验证集泄露。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("lung_nodule.csv")
X = df[["Age", "NoduleSize", "Spiculation", "Smoking", "RadiomicsScore"]]
y = df["Malignant"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

fit = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ("svm", SVC(probability=True, random_state=42))
    ],
    voting="soft",
    weights=[1, 2, 1]
)
fit.fit(X_train, y_train)

pred_prob = fit.predict_proba(X_test)[:, 1]
```

### 5.2 R

常用包：

- `caret`
- `caretEnsemble`

```r
library(caret)
library(caretEnsemble)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

models <- caretList(
  Malignant ~ Age + NoduleSize + Spiculation + Smoking + RadiomicsScore,
  data = df_train,
  trControl = ctrl,
  methodList = c("glm", "rf")
)

ens <- caretEnsemble(models, metric = "ROC")
pred <- predict(ens, newdata = df_test)
```

## 6. 结果如何解释

- 核心结果看什么：集成模型是否优于最佳单模型、概率校准、不同阈值下的灵敏度和特异度。
- 每个主要参数如何解释：权重越高，该模型对最终概率影响越大；软投票更依赖概率质量。
- 临床或医学意义如何表达：可表达为“综合多个模型后得到的风险概率”，但仍需外部验证。
- 常见误读：投票集成不是因果证据，也不自动保证比最佳单模型更好。

## 7. 推荐可视化

- 单模型与集成模型 ROC / PR 曲线对比。
- 校准曲线。
- 各模型预测概率的相关性热图。

## 8. 优势、局限与常见坑

### 优势

- 思路简单、实现成本低。
- 可以快速融合不同模型优势。
- 基模型可并行训练。

### 局限

- 不学习复杂的模型间交互。
- 软投票依赖概率校准。
- 权重选择容易过拟合验证集。

### 常见坑

- 用测试集调投票权重。
- 把未校准模型概率直接相加。
- 基模型高度相似，导致集成收益有限。

## 9. 与相近方法的区别

- 和 [[Stacking集成（Stacked Generalization）]] 的区别：Stacking 用元学习器学习如何融合预测，投票集成通常只做固定规则合并。
- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：Bagging 自动从重抽样数据训练多个同类模型；投票集成常融合不同模型。
- 和 [[Boosting算法（Boosting）]] 的区别：Boosting 是串行纠错，投票集成没有逐轮纠错过程。

## 10. 医学研究中的典型应用

- 融合临床变量模型、影像组学模型和实验室指标模型。
- 多中心模型集成，降低单中心模型偏差。
- 对同一分类任务进行稳健模型平均。

## 11. 相关方法

- [[Stacking集成（Stacked Generalization）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[Boosting算法（Boosting）]]
- [[随机森林（Random Forest）]]

## 12. 参考资料

- Dietterich TG. Ensemble methods in machine learning. In: *Multiple Classifier Systems*. 2000:1-15.
- Zhou ZH. *Ensemble Methods: Foundations and Algorithms*. Chapman and Hall/CRC; 2012.
- scikit-learn Developers. `sklearn.ensemble.VotingClassifier`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) （访问日期：2026-07-02）

---
title: 线性判别分析
english_name: Linear Discriminant Analysis, LDA
slug: linear-discriminant-analysis-lda
aliases: [LDA, linear discriminant analysis, Fisher discriminant analysis, "线性判别分析（Linear Discriminant Analysis, LDA）"]
category: 分类与判别方法
subcategory: 判别分析
tags: [医学统计, 数据科学, 分类, 判别分析, 监督式降维]
status: 已建
difficulty: intermediate
question_type: 分类判别与监督式线性降维
data_type: [表格数据, 连续特征矩阵]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn]
r_packages: [MASS]
---

# 线性判别分析（Linear Discriminant Analysis, LDA）

## 1. 方法概览

### 1.1 定义

线性判别分析是一种经典的监督式分类和降维方法。它寻找能让组间差异尽量大、组内差异尽量小的线性投影方向，并可在这些方向上进行分类。

### 1.2 它主要解决什么问题

- 研究问题：多组样本能否通过连续特征的线性组合被区分。
- 适用任务：二分类、多分类、监督式降维、判别方向可视化。
- 常见医学场景：良恶性分类，疾病亚型判别，基于生物标志物组合的风险分组。

### 1.3 直觉理解

LDA 希望找到一条或几条轴，使不同类别的中心在投影后尽量分开，同时同一类别内部尽量聚在一起。它既可以作为分类器，也可以把高维特征压缩成最多 $C-1$ 个判别维度。

## 2. 数学形式

### 2.1 核心公式

对于类别 $c=1,\dots,C$，设类别均值为 $\mu_c$，总体均值为 $\mu$。类内散布矩阵：

$$
S_W=\sum_{c=1}^{C}\sum_{x_i\in c}(x_i-\mu_c)(x_i-\mu_c)^\top
$$

类间散布矩阵：

$$
S_B=\sum_{c=1}^{C}n_c(\mu_c-\mu)(\mu_c-\mu)^\top
$$

LDA 的投影方向 $w$ 最大化 Fisher 判别准则：

$$
J(w)=\frac{w^\top S_B w}{w^\top S_W w}
$$

多维情况下求解广义特征值问题：

$$
S_B w = \lambda S_W w
$$

在生成式分类视角下，LDA 假设各类别服从共享协方差矩阵的多元正态分布：

$$
x\mid y=c \sim N(\mu_c,\Sigma)
$$

### 2.2 参数或统计量含义

- $S_W$：类内散布，衡量同类样本内部变异。
- $S_B$：类间散布，衡量不同类别中心的分离程度。
- $w$：判别方向或投影向量。
- `n_components`：保留的判别维度，最多为类别数减 1。
- 先验概率：各类别在总体中的预期比例。
- 共享协方差矩阵：LDA 分类假设中各类别共同的协方差结构。

### 2.3 关键假设

- 类别条件分布近似多元正态。
- 各类别协方差矩阵近似相同。
- 特征与类别之间的区分结构可由线性边界表达。
- 样本量足以估计协方差矩阵，类别不平衡需谨慎处理。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量或标准化后的数值特征。
- 因变量类型：二分类或多分类标签。
- 数据结构：监督学习特征矩阵加类别标签。
- 是否适合高维数据：特征数接近或超过样本数时普通 LDA 不稳定，需正则化或先降维。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：普通 LDA 假设样本独立，重复测量需先汇总或使用其他模型。

### 3.2 示例表格

以肿瘤良恶性判别为例：

| Radius | Texture | Smoothness | Compactness | Diagnosis |
| --- | --- | --- | --- | --- |
| 17.9 | 10.4 | 0.118 | 0.278 | malignant |
| 12.3 | 14.5 | 0.090 | 0.080 | benign |
| 20.5 | 21.4 | 0.109 | 0.160 | malignant |
| 11.8 | 17.1 | 0.082 | 0.060 | benign |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和类别标签。
- 关键变量：类别先验、solver、是否使用 shrinkage、保留判别维度。
- 需要预处理的内容：缺失处理、异常值检查、标准化、训练测试划分、类别不平衡评估。

#### 产出

- 模型对象/统计结果：判别方向、类别均值、共享协方差估计、类别预测概率。
- 参数估计：判别系数、类别先验、协方差矩阵。
- 预测结果：类别标签和类别概率。
- 不确定性指标：交叉验证性能、混淆矩阵、置信区间或 bootstrap 稳定性。

## 4. 适用场景

- 适合：类别较清晰、特征连续、希望得到线性判别方向和可解释分类边界的场景。
- 不适合：类别协方差差异很大、边界明显非线性、特征数远大于样本数且未正则化的场景。
- 使用前需要特别检查的点：协方差齐性、类别不平衡、异常值、训练测试信息泄露。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("tumor_features.csv")
X = df[["Radius", "Texture", "Smoothness", "Compactness"]]
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis(n_components=1)
)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))
print(model.predict_proba(X_test).round(3)[:5])
```

### 5.2 R

常用包：

- `MASS`

```r
library(MASS)

fit <- lda(
  Diagnosis ~ Radius + Texture + Smoothness + Compactness,
  data = train_df
)

pred <- predict(fit, newdata = test_df)
table(Predicted = pred$class, Observed = test_df$Diagnosis)
head(pred$posterior)
```

## 6. 结果如何解释

- 核心结果看什么：判别方向、类别均值分离、预测概率、混淆矩阵和交叉验证性能。
- 每个主要参数如何解释：判别系数表示原始特征在线性判别方向中的贡献，但需结合标准化和变量相关性解释。
- 临床或医学意义如何表达：LDA 给出的是特征组合的判别规则，应同时报告区分度、误分类情况和适用人群。
- 常见误读：LDA 的投影方向能分开样本，不代表每个单变量都有独立临床效应。

## 7. 推荐可视化

- LD1 或 LD1-LD2 判别得分散点图。
- 各类别判别得分密度图。
- 混淆矩阵热图。
- ROC 曲线和校准图，尤其用于二分类医学预测。

## 8. 优势、局限与常见坑

### 优势

- 模型简洁，计算快速。
- 判别方向可用于监督式降维和可视化。
- 在正态且协方差相近时表现稳定。

### 局限

- 线性边界表达能力有限。
- 对协方差齐性和异常值敏感。
- 高维小样本时协方差矩阵估计不稳定。

### 常见坑

- 把 LDA 与主题模型 Latent Dirichlet Allocation 混淆。
- 在标准化、特征选择或降维中使用全数据导致信息泄露。
- 忽略类别不平衡，只报告 accuracy。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 不使用类别标签，保留最大方差方向；LDA 使用类别标签，寻找最大判别方向。
- 和 [[Logistic回归（Logistic Regression）]] 的区别：Logistic 回归直接建模类别概率，LDA 是带正态和共享协方差假设的生成式判别模型。
- 和 [[朴素贝叶斯（Naive Bayes）]] 的区别：朴素贝叶斯常假设特征条件独立，LDA 允许特征相关但要求类别共享协方差。

## 10. 医学研究中的典型应用

- 基于连续生物标志物组合的疾病分类。
- 肿瘤良恶性或分子亚型判别。
- 在少数判别维度上可视化多类别临床表型。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Logistic回归（Logistic Regression）]]
- [[朴素贝叶斯（Naive Bayes）]]
- [[支持向量机（Support Vector Machine, SVM）]]

## 12. 参考资料

- Fisher RA. The use of multiple measurements in taxonomic problems. *Annals of Eugenics*. 1936;7(2):179-188.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- scikit-learn Developers. `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) （访问日期：2026-07-02）

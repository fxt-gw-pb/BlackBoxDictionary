---
title: 随机森林回归
english_name: Random Forest Regression
aliases: [random forest regression, RF regression, "随机森林回归（Random Forest Regression）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 树模型]
status: draft
difficulty: basic
question_type: 连续结局集成建模
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scikit-learn]
r_packages: [randomForest]
related_methods: [随机森林, 决策树回归, 梯度提升回归]
---

# 随机森林回归（Random Forest Regression）

## 1. 方法概览

### 1.1 定义

随机森林回归通过训练多棵带有随机性的回归树，并对它们的预测结果取平均，从而获得更稳定、更鲁棒的连续结局预测模型。

### 1.2 它主要解决什么问题

- 研究问题：如何在复杂非线性和多交互场景下获得更稳健的回归预测。
- 适用任务：连续结局预测、非线性建模、变量重要性评估。
- 常见医学场景：由多种临床变量预测连续风险评分、费用、住院时长或实验室指标。

### 1.3 直觉理解

随机森林可以理解为“让很多棵不同视角的回归树各自给出判断”，然后把这些判断平均起来，减少单棵树的波动和过拟合。

## 2. 数学形式

### 2.1 核心公式

若共有 $B$ 棵回归树，则随机森林的预测为：

$$
\hat f(x)=\frac{1}{B}\sum_{b=1}^{B} T_b(x)
$$

其中 $T_b(x)$ 是第 $b$ 棵树对样本 $x$ 的预测。

### 2.2 参数或统计量含义

- Bootstrap 采样：每棵树看到的是带放回抽样后的训练集。
- 随机特征子集：每次分裂时只在部分特征中找最佳分裂。
- $B$：树的数量，越多通常越稳定但计算更慢。

### 2.3 关键假设

- 对函数形式没有强假设。
- 依赖多树平均降低方差。
- 特征空间中存在可被树结构捕捉的分裂模式。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、分类、离散变量均可。
- 因变量类型：连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：可以，但高维稀疏场景并非最优。
- 是否适合缺失较多数据：通常先做缺失处理更稳妥。
- 是否适合删失数据：不适合。
- 是否适合重复测量数据：不直接适合。

### 3.2 示例表格

随机森林回归常见的数据结构与普通表格预测任务一致：

| OverallQual | GrLivArea | GarageCars | TotalBsmtSF | YearBuilt | SalePrice |
| --- | --- | --- | --- | --- | --- |
| 7 | 1710 | 2 | 856 | 2003 | 208500 |
| 6 | 1262 | 2 | 1262 | 1976 | 181500 |
| 7 | 1786 | 2 | 920 | 2001 | 223500 |
| 7 | 1717 | 3 | 756 | 1915 | 140000 |
| 8 | 2198 | 3 | 1145 | 2000 | 250000 |

### 3.3 输入与产出

#### 输入

- 输入数据：连续结局和特征矩阵。
- 关键变量：树数 `n_estimators`、树深度、最小分裂样本数、最大特征数。
- 需要预处理的内容：缺失处理、训练测试集划分。

#### 产出

- 模型对象/统计结果：多棵树的集成模型、OOB 误差（可选）、特征重要性。
- 参数估计：树结构本身不直接解释，更多看重要性排序。
- 预测结果：连续型预测值。
- 不确定性指标：交叉验证误差、测试集 MSE、$R^2$。

## 4. 适用场景

- 适合：表格数据、非线性关系、交互关系复杂、对预测性能要求较高的场景。
- 不适合：强解释导向、希望得到简单系数模型时。
- 使用前需要特别检查的点：特征重要性偏倚、调参范围、训练成本。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
from sklearn.ensemble import RandomForestRegressor

fit = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
fit.fit(X_train, y_train)
y_pred = fit.predict(X_test)
```

### 5.2 R

常用包：

- `randomForest`

```r
library(randomForest)

fit <- randomForest(SalePrice ~ ., data = df, ntree = 500, importance = TRUE)
pred <- predict(fit, newdata = df_test)
importance(fit)
```

## 6. 结果如何解释

- 核心结果看什么：预测误差、稳定性、特征重要性。
- 每个主要参数如何解释：随机森林通常更偏预测解释，而不是参数解释。
- 临床或医学意义如何表达：适合表达“哪些变量最能区分或预测结果”，但不适合直接做线性效应解读。
- 常见误读：特征重要性高不等于因果重要。

## 7. 推荐可视化

- 特征重要性条形图。
- 真实值 vs 预测值散点图。
- 局部依赖图或 SHAP（后续可扩展）。

### 7.1 图像示例

下图给出随机森林回归在房价数据上的特征重要性排序，适合说明模型关注的主要变量。

![](../../04_示例图像/random_forest_feature_importance.png)

## 8. 优势、局限与常见坑

### 优势

- 非线性与交互建模能力强。
- 对异常值和噪声较稳健。
- 一般比单棵树泛化更好。

### 局限

- 解释性弱于单棵树和线性模型。
- 训练和调参成本更高。
- 重要性指标可能偏向连续变量或高基数变量。

### 常见坑

- 只看重要性排序，不做外部验证。
- 把高重要性变量误解成因果变量。
- 树数太少导致结果不稳定。

## 9. 与相近方法的区别

- 和决策树回归的区别：随机森林通过 bagging 多树集成来降方差。
- 和随机森林的区别：这里聚焦连续结局；二分类或多分类任务看主条目“随机森林”。
- 和梯度提升回归的区别：随机森林是并行 bagging，梯度提升是串行 boosting。
- 和线性回归的区别：随机森林不依赖线性假设。

## 10. 医学研究中的典型应用

- 表格型临床预测模型。
- 多变量连续结局风险建模。
- 特征重要性初筛。

## 11. 相关方法

- [[随机森林（Random Forest）]]
- [[决策树回归（Decision Tree Regression）]]
- [[梯度提升回归（Gradient Boosting Regression）]]
- [[支持向量回归（Support Vector Regression, SVR）]]

## 12. 参考资料

- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- scikit-learn Developers. `sklearn.ensemble.RandomForestRegressor`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) （访问日期：2026-07-02）
- CRAN. Package `randomForest`. [https://cran.r-project.org/package=randomForest](https://cran.r-project.org/package=randomForest) （访问日期：2026-07-02）

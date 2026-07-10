---
title: 多项式核回归
english_name: Polynomial Kernel Regression
slug: polynomial-kernel-regression
aliases: [polynomial kernel regression, SVR with polynomial kernel, "多项式核回归（Polynomial Kernel Regression）"]
category: 支持向量机与核方法
subcategory: 多项式核方法
tags: [医学统计, 数据科学, 核方法, 支持向量机]
status: 已建
difficulty: intermediate
question_type: 非线性核回归
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scikit-learn]
r_packages: [e1071]
---

# 多项式核回归（Polynomial Kernel Regression）

## 1. 方法概览

### 1.1 一句话本质

多项式核回归通过计算多项式核内积，隐式使用幂次项和交互项，而无需把高维多项式特征逐列展开。

### 1.2 定义

多项式核回归泛指使用多项式核的核回归方法。本卡聚焦实践中最常见的形式：带多项式核的 [[支持向量回归（Support Vector Regression, SVR）]]。

### 1.3 它主要解决什么问题

- 连续结局存在低阶弯曲或变量间乘积交互。
- 显式多项式展开会迅速增加特征数。
- 希望保留 SVR 的 $\epsilon$-不敏感损失和正则化。

### 1.4 直觉与类比

二次多项式不只看 $x_1,x_2$，还看 $x_1^2,x_2^2,x_1x_2$。核函数像一台内积计算器：不把这些新列真的建出来，也能得到它们在扩展空间中的内积。

## 2. 核心思想与原理

### 2.1 显式展开的困难

$p$ 个变量做 $d$ 阶展开时，特征数可快速增长。核方法用 $K(x,z)=\phi(x)^\top\phi(z)$ 直接计算高维内积，避免显式存储 $\phi(x)$。

### 2.2 阶数决定结构

二次核允许二次曲率和两两交互；更高阶核允许更复杂结构，也更易数值不稳定和过拟合。阶数不是越高越好。

### 2.3 与普通多项式回归不同

普通多项式回归通常用平方损失并直接估计展开特征系数；多项式核 SVR 使用间隔损失、对偶表示和支持向量，单个幂次项的系数不直接可见。

## 3. 数学形式

### 3.1 多项式核

$$
K(x,z)=(\gamma x^\top z+c_0)^d
$$

其中 $d$ 为阶数，$\gamma$ 控制内积缩放，$c_0$ 控制低阶项在核中的贡献。

### 3.2 核预测函数

$$
f(x)=
\sum_{i=1}^{n}\beta_iK(x_i,x)+b
$$

在 SVR 中，$\beta_i=\alpha_i-\alpha_i^*$，仅支持向量对应的 $\beta_i$ 通常非零。

### 3.3 二次核的隐式特征

当 $\gamma=1,c_0=1,d=2$ 且 $x=(x_1,x_2)$ 时，可取：

$$
\phi(x)=
(x_1^2,\sqrt2x_1x_2,x_2^2,
\sqrt2x_1,\sqrt2x_2,1)
$$

于是 $\phi(x)^\top\phi(z)=(x^\top z+1)^2$。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 多项式结构合理 | RBF 或树模型可能更合适 | 比较验证性能 |
| 特征已标准化 | 高次幂放大量纲差异 | 管线内缩放 |
| 阶数较低且正则充分 | 边界震荡、过拟合 | 嵌套调参 |
| 样本规模适中 | 核矩阵成本过高 | 与显式近似比较 |

## 4. 手把手算例

取二次核 $K(x,z)=(x^\top z+1)^2$，两个二维向量：

$$
x=(1,2),\qquad z=(2,1)
$$

**Step 1：直接用核。**

$$
x^\top z=1(2)+2(1)=4
$$

$$
K(x,z)=(4+1)^2=25
$$

**Step 2：显式展开验证。**

$$
\phi(x)=(1,2\sqrt2,4,\sqrt2,2\sqrt2,1)
$$

$$
\phi(z)=(4,2\sqrt2,1,2\sqrt2,\sqrt2,1)
$$

两者内积为：

$$
4+8+4+4+4+1=25
$$

与核函数完全一致。

**Step 3：放入回归预测。** 再考虑一维支持向量 $x_1=1,x_2=2$，系数 $\beta_1=0.5,\beta_2=-0.2$，截距 $b=0.1$。预测 $x=1.5$：

$$
K(1,1.5)=6.25,\qquad K(2,1.5)=16
$$

$$
f(1.5)=0.5(6.25)-0.2(16)+0.1=0.025
$$

**结论：** 模型只需核值和支持向量系数，就能使用隐式多项式特征完成预测。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续结局，特征通常先标准化。
- 类别变量需妥善编码；缺失需预先处理。
- 普通方法不直接处理删失或重复测量相关性。

### 5.2 输入与产出

输入包括 $d,\gamma,c_0,C,\epsilon$。输出为连续预测、支持向量和对偶系数，不直接输出各显式幂次项的可解释回归系数。

## 6. 适用场景

- 领域知识提示低阶多项式曲率或交互。
- 特征数不大、样本规模适中。
- 不适合超大样本、极高阶结构或需要逐项解释多项式系数的研究。

## 7. 实现

### 7.1 Python

```python
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

pipe = make_pipeline(
    StandardScaler(),
    SVR(kernel="poly")
)
grid = {
    "svr__degree": [2, 3],
    "svr__C": [0.1, 1, 10],
    "svr__epsilon": [0.05, 0.1, 0.2],
    "svr__gamma": ["scale", 0.1],
    "svr__coef0": [0, 1],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv)
search.fit(X_train, y_train)
pred = search.predict(X_test)
print(search.best_params_)
```

### 7.2 R

```r
library(e1071)

fit <- svm(
  y ~ .,
  data = train,
  type = "eps-regression",
  kernel = "polynomial",
  degree = 2,
  gamma = 0.1,
  coef0 = 1,
  cost = 10,
  epsilon = 0.1,
  scale = TRUE
)

pred <- predict(fit, newdata = test)
```

## 8. 结果如何解释

- `degree` 决定允许的最高交互阶数，但不代表所有高阶项都同等重要。
- `coef0` 越大，低阶项在非齐次多项式核中的贡献越明显。
- 整体预测曲线可解释，单个原始变量的独立效应不容易从对偶模型直接读出。
- 拟合关系是预测关联，不应直接称为剂量的因果效应。

## 9. 诊断与稳健性

1. 比较线性、二次、三次与 RBF 核的嵌套验证性能。
2. 画阶数和 $C$ 对验证误差的影响。
3. 检查支持向量比例及训练时间。
4. 与显式二次回归比较，判断复杂核是否真正增益。
5. 检查训练范围外的曲线和极端预测。

## 10. 推荐可视化

- 原始散点与多项式核拟合曲线。
- 不同阶数的验证曲线。
- 真实值-预测值和残差图。
- 二维时可画预测曲面。

下图展示多项式核回归的非线性拟合：

![](../../04_示例图像/polynomial_kernel_regression_curve.png)

## 11. 优势、局限与常见坑

**优势：** 隐式使用多项式和交互特征，可与 SVR 正则化和容忍损失结合。

**局限：** 阶数和尺度敏感，可解释性弱，大样本训练成本高。

**常见坑：** 不标准化；阶数过高；忽略 `gamma` 和 `coef0`；把它等同于普通多项式回归；仅凭训练曲线选阶数。

## 12. 与相近方法的区别

- [[多项式回归（Polynomial Regression）]]：显式展开并可解释各项系数。
- [[支持向量回归（Support Vector Regression, SVR）]]：多项式核回归是 SVR 的一个核选择。
- [[高斯过程回归（Gaussian Process Regression）]]：同样使用核，但 GPR 给出后验预测分布。
- 选择经验：若低阶项解释重要，优先显式多项式回归；若预测优先且特征展开庞大，可考虑核方法。

## 13. 医学研究中的典型应用

- 预测性剂量-反应和生理指标曲线。
- 低维生物标志物间交互的连续结局预测。
- 小样本影像或组学特征的非线性回归。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| polynomial kernel | 隐式表示多项式特征内积的核 |
| degree | 多项式核的阶数 |
| `coef0` | 控制低阶与高阶项相对影响的常数 |
| feature map | 从原空间到隐式特征空间的映射 |
| kernel matrix | 所有样本两两核相似度组成的矩阵 |

## 15. 相关方法

- [[支持向量回归（Support Vector Regression, SVR）]]
- [[多项式回归（Polynomial Regression）]]
- [[高斯过程回归（Gaussian Process Regression）]]
- [[支持向量机（Support Vector Machine, SVM）]]

## 16. 参考资料

- Schölkopf B, Smola AJ. *Learning with Kernels*. MIT Press; 2002.
- Smola AJ, Schölkopf B. A tutorial on support vector regression. *Stat Comput*. 2004;14:199-222.
- scikit-learn Developers. `SVR` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) （访问日期：2026-07-09）

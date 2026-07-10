---
title: 支持向量回归
english_name: Support Vector Regression, SVR
slug: support-vector-regression-svr
aliases: [SVR, support vector regression, "支持向量回归（Support Vector Regression, SVR）"]
category: 支持向量机与核方法
subcategory: 支持向量回归
tags: [医学统计, 数据科学, 支持向量机, 回归分析]
status: 已建
difficulty: intermediate
question_type: 连续结局非线性建模
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scikit-learn]
r_packages: [e1071]
---

# 支持向量回归（Support Vector Regression, SVR）

## 1. 方法概览

### 1.1 一句话本质

SVR 寻找尽可能平坦的回归函数，并允许预测曲线周围存在一个不计损失的 $\epsilon$ 容忍管。

### 1.2 定义

支持向量回归是 [[支持向量机（Support Vector Machine, SVM）]] 的连续结局版本。它不最小化所有点的平方误差，而用 $\epsilon$-不敏感损失忽略足够小的误差，只惩罚超出容忍管的部分。

### 1.3 它主要解决什么问题

- 小到中等样本中的非线性连续结局预测。
- 允许存在临床上可忽略误差的稳健函数拟合。
- 通过核函数在不显式构造高维特征时学习弯曲关系。

### 1.4 直觉与类比

先在预测曲线周围画一条宽度为 $2\epsilon$ 的管道。管内点已经“够准”，不再消耗模型精力；管边界和管外点决定曲线的位置与形状。

## 2. 核心思想与原理

### 2.1 平坦与拟合的权衡

$\|w\|^2/2$ 鼓励函数平坦，超出 $\epsilon$ 管的松弛量鼓励贴近数据，$C$ 决定后者相对前者有多重要。

### 2.2 稀疏表示

对预测函数有非零对偶系数的训练点称为支持向量。管内且不在边界上的点通常不直接出现在最终核展开中，因此模型由部分关键病例决定。

### 2.3 核技巧

线性 SVR 在原特征空间拟合平面；RBF 或多项式核把内积替换为相似度，使线性算法可在隐式高维空间形成非线性曲线。

## 3. 数学形式

### 3.1 原始优化问题

$$
\operatorname*{minimize}_{w,b,\xi,\xi^*}
\quad
\frac12\|w\|^2
+C\sum_{i=1}^{n}(\xi_i+\xi_i^*)
$$

约束为：

$$
\begin{aligned}
y_i-w^\top\phi(x_i)-b &\le \epsilon+\xi_i\\
w^\top\phi(x_i)+b-y_i &\le \epsilon+\xi_i^*\\
\xi_i,\xi_i^* &\ge 0
\end{aligned}
$$

### 3.2 $\epsilon$-不敏感损失

$$
L_\epsilon(y,f)=
\max\{0,|y-f(x)|-\epsilon\}
$$

### 3.3 核化预测

$$
f(x)=
\sum_{i=1}^{n}(\alpha_i-\alpha_i^*)K(x_i,x)+b
$$

常用 RBF 核为：

$$
K(x,z)=\exp[-\gamma\|x-z\|^2]
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 特征已合理缩放 | 大量纲变量支配核距离 | 管线内标准化 |
| $C,\epsilon,\gamma$ 经训练内调参 | 欠拟合或过拟合 | 嵌套交叉验证 |
| 样本规模适中 | 核矩阵训练成本高 | 记录时间与内存 |
| 数据切分无患者泄漏 | 测试误差虚低 | 分组或时间切分 |

## 4. 手把手算例

假设候选线性函数为 $f(x)=x$，4 个观测为：

| $x$ | $y$ | $f(x)$ | 绝对误差 |
| ---: | ---: | ---: | ---: |
| 0 | 0.1 | 0 | 0.1 |
| 1 | 1.1 | 1 | 0.1 |
| 2 | 2.4 | 2 | 0.4 |
| 3 | 4.0 | 3 | 1.0 |

取 $\epsilon=0.2$。各点的 $\epsilon$-不敏感损失为：

$$
(0,\ 0,\ 0.4-0.2,\ 1.0-0.2)
=(0,0,0.2,0.8)
$$

前两点位于管内，不计损失；后两点超出管道，是决定修正方向的支持病例。

若 $C=2$，总误差惩罚为：

$$
C\sum_iL_\epsilon=2(0.2+0.8)=2
$$

又因为 $w=1$，平坦度惩罚为 $\|w\|^2/2=0.5$，该候选函数的目标值为 $2.5$。

**参数直觉：** 增大 $\epsilon$ 会让更多点落入管内；增大 $C$ 会更强烈地要求模型修正管外误差。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 因变量为连续型。
- 特征可连续或经编码后进入模型，通常必须标准化。
- 缺失需预先处理；普通 SVR 不直接处理删失或重复测量相关性。

### 5.2 输入与产出

输入为特征、连续结局、核函数及 $C,\epsilon,\gamma$。输出为连续预测、支持向量和对偶系数。标准实现不自然给出预测区间。

## 6. 适用场景

- 小到中等样本、特征数中等、非线性明显的连续预测。
- 误差在某个容忍范围内临床意义相近的任务。
- 不适合超大样本、要求参数效应解释或必须有可靠预测区间的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.05, 0.1, 0.2],
    "svr__gamma": ["scale", 0.01, 0.1],
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv)
search.fit(X_train, y_train)

pred = search.predict(X_test)
print(search.best_params_)
print(mean_absolute_error(y_test, pred))
```

### 7.2 R

```r
library(e1071)

tuned <- tune.svm(
  y ~ .,
  data = train,
  type = "eps-regression",
  kernel = "radial",
  cost = c(0.1, 1, 10, 100),
  epsilon = c(0.05, 0.1, 0.2),
  gamma = c(0.01, 0.1)
)

fit <- tuned$best.model
pred <- predict(fit, newdata = test)
mean(abs(test$y - pred))
```

## 8. 结果如何解释

- MAE 与 $\epsilon$ 都有结局原单位，应结合临床可接受误差说明。
- `C` 越大越不容忍管外误差；$\gamma$ 越大，RBF 影响范围越局部。
- 支持向量比例过高可能提示管太窄、噪声大或模型复杂。
- 预测曲线是条件预测，不代表暴露的因果剂量反应。

## 9. 诊断与稳健性

1. 比较不同 $C,\epsilon,\gamma$ 的验证误差曲面。
2. 检查支持向量比例与残差分布。
3. 与线性回归、Ridge 和树模型建立基准。
4. 做重复嵌套交叉验证，避免调参乐观偏倚。
5. 检查训练范围外预测、时间外与中心外泛化。

## 10. 推荐可视化

- 散点、SVR 曲线和 $\epsilon$ 管。
- 真实值-预测值与残差图。
- $C,\epsilon,\gamma$ 的验证性能热图。

下图展示非线性 SVR 拟合曲线：

![](../../04_示例图像/svr_nonlinear_curve.png)

下图展示真实值与预测值：

![](../../04_示例图像/svr_actual_vs_pred_houseprice.png)

## 11. 优势、局限与常见坑

**优势：** 非线性灵活，对管内小误差不敏感，解由支持向量稀疏表示。

**局限：** 调参敏感，大样本训练慢，不直接提供概率型不确定性。

**常见坑：** 未标准化；只调 $C$；调参和评估共用同一折；把 $\epsilon$ 当置信区间；在数万以上样本直接使用精确 RBF-SVR。

## 12. 与相近方法的区别

- [[支持向量机（Support Vector Machine, SVM）]]：SVM 最大化分类间隔，SVR 构造回归容忍管。
- [[高斯过程回归（Gaussian Process Regression）]]：GPR 是概率模型并自然输出不确定性。
- [[多项式核回归（Polynomial Kernel Regression）]]：是 SVR 选用多项式核时的具体形式。
- [[随机森林回归（Random Forest Regression）]]：树更擅长局部阈值与交互，SVR 更偏平滑核函数。

## 13. 医学研究中的典型应用

- 连续生理指标、风险评分与资源消耗预测。
- 小样本影像组学或组学连续结局建模。
- 非线性剂量或暴露-响应的预测性拟合。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| $\epsilon$-tube | 预测曲线周围不计损失的容忍带 |
| support vector | 位于管边界或管外并决定解的训练点 |
| slack variable | 超出容忍带的误差量 |
| kernel trick | 通过核内积隐式进入高维特征空间 |
| RBF kernel | 相似度随平方距离指数衰减的核 |

## 15. 相关方法

- [[支持向量机（Support Vector Machine, SVM）]]
- [[高斯过程回归（Gaussian Process Regression）]]
- [[多项式核回归（Polynomial Kernel Regression）]]
- [[Ridge回归（Ridge Regression）]]
- [[随机森林回归（Random Forest Regression）]]

## 16. 参考资料

- Smola AJ, Schölkopf B. A tutorial on support vector regression. *Stat Comput*. 2004;14:199-222.
- scikit-learn Developers. `SVR` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) （访问日期：2026-07-09）
- CRAN. Package `e1071`. [https://cran.r-project.org/package=e1071](https://cran.r-project.org/package=e1071) （访问日期：2026-07-09）

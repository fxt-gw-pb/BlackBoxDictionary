---
title: 支持向量机
english_name: Support Vector Machine, SVM
slug: support-vector-machine-svm
aliases: [SVM, support vector machine, "支持向量机（Support Vector Machine, SVM）"]
category: 支持向量机与核方法
subcategory: 间隔最大化分类
tags: [医学统计, 数据科学, 支持向量机, 分类]
status: 已建
difficulty: intermediate
question_type: 间隔最大化分类
data_type: [表格数据]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn]
r_packages: [e1071]
---

# 支持向量机（Support Vector Machine, SVM）

## 1. 方法概览

### 1.1 一句话本质

SVM 不只寻找能分开两类的边界，还寻找离最近训练点尽可能远的边界，并允许用惩罚换取少量违例。

### 1.2 定义

支持向量机是最大间隔监督分类方法。线性 SVM 在原空间寻找超平面；核 SVM 通过核函数在隐式特征空间寻找线性边界，从原空间看可形成非线性决策面。

### 1.3 它主要解决什么问题

- 小到中等样本、高维特征的二分类。
- 类别边界复杂但可由核相似度表达的任务。
- 希望让边界主要由最困难的邻界病例决定。

### 1.4 直觉与类比

两类病例之间可能有很多分割线。SVM 选择中间“走廊”最宽的一条；最贴近走廊边缘的病例是支持向量，远处病例移动一点通常不会改变边界。

## 2. 核心思想与原理

### 2.1 最大间隔为何有用

在可分数据中，仅要求训练集零错误会得到很多边界。最大化最近点到边界的距离，相当于选择对小扰动更稳健的分隔面。

### 2.2 软间隔

医学数据通常不可完全分开。松弛变量允许病例落入间隔甚至被错分，$C$ 控制违例代价。大 $C$ 更重视训练误差，小 $C$ 更重视宽间隔。

### 2.3 核化

决策只依赖训练点内积。将内积替换为核 $K(x_i,x)$，就能在不显式映射的情况下学习非线性边界。

## 3. 数学形式

### 3.1 软间隔原始问题

设 $y_i\in\{-1,+1\}$：

$$
\operatorname*{minimize}_{w,b,\xi}
\quad
\frac12\|w\|^2+C\sum_{i=1}^{n}\xi_i
$$

满足：

$$
y_i(w^\top x_i+b)\ge1-\xi_i,\qquad \xi_i\ge0
$$

### 3.2 间隔与 hinge loss

规范化后，点到超平面的单侧间隔为：

$$
\frac{1}{\|w\|}
$$

两条支持超平面之间的完整宽度为 $2/\|w\|$。hinge loss 为：

$$
L(y,f)=\max[0,1-yf(x)]
$$

### 3.3 核决策函数

$$
f(x)=
\sum_{i=1}^{n}\alpha_iy_iK(x_i,x)+b
$$

$$
\hat y=\operatorname{sign}[f(x)]
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 特征合理缩放 | 大量纲变量支配距离 | 管线内标准化 |
| 核与超参数适当 | 欠拟合或边界过度曲折 | 嵌套交叉验证 |
| 类别与采样机制明确 | 准确率掩盖少数类失败 | PR、敏感度、类别权重 |
| 样本规模可计算 | 核 SVM 训练过慢 | 线性 SVM 或核近似 |

## 4. 手把手算例

一维数据有 4 个点：

| $x$ | $y$ |
| ---: | ---: |
| -2 | -1 |
| -1 | -1 |
| 1 | +1 |
| 2 | +1 |

选择决策函数 $f(x)=wx+b=x$，即 $w=1,b=0$。

**Step 1：检查约束。**

四点的函数间隔 $y_if(x_i)$ 分别为：

$$
(2,1,1,2)
$$

均至少为 1，所以无需松弛变量。

**Step 2：找支持向量。** $x=-1$ 与 $x=1$ 恰好满足 $y_if(x_i)=1$，它们是支持向量；$x=-2,2$ 距边界更远。

**Step 3：算间隔。**

$$
\text{单侧间隔}=\frac{1}{|w|}=1
$$

$$
\text{完整间隔宽度}=\frac{2}{|w|}=2
$$

决策边界为 $x=0$，支持超平面为 $x=-1$ 与 $x=1$。

**Step 4：加入违例。** 若新增负类点 $x=0.2,y=-1$，则 $yf(x)=-0.2$：

$$
\xi\ge1-(-0.2)=1.2
$$

该点被错分且深入错误侧。大 $C$ 会强迫边界迁就它，小 $C$ 更可能接受这次违例以保留宽间隔。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 二分类最自然；多分类通常由 one-vs-one 或 one-vs-rest 组合。
- 连续或向量化特征通常需标准化。
- 缺失需预先处理；普通 SVM 不直接处理删失或重复测量。

### 5.2 输入与产出

输入为特征、类别、核、$C$、$\gamma$ 和类别权重。输出为类别、决策分数、支持向量及可选概率。SVM 本身是判别边界模型，不是天然概率模型。

## 6. 适用场景

- 高维小中样本，如组学、影像组学和文本向量。
- 类别边界相对清晰或核映射后可分。
- 不适合超大样本、需直接系数解释或必须快速更新的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, brier_score_loss

base = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,
    ),
)
model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob))
print(brier_score_loss(y_test, prob))
```

### 7.2 R

```r
library(e1071)

fit <- svm(
  TumorSubtype ~ .,
  data = train,
  kernel = "radial",
  cost = 1,
  gamma = 0.1,
  class.weights = c(A = 1, B = 2),
  scale = TRUE,
  probability = TRUE
)

pred <- predict(fit, newdata = test, probability = TRUE)
prob <- attr(pred, "probabilities")
```

## 8. 结果如何解释

- 决策函数正负决定类别，绝对值表示离边界的函数尺度，不是概率。
- $C$ 越大通常边界越迁就训练点；RBF 的 $\gamma$ 越大，单点影响越局部。
- 支持向量多说明更多病例参与定义边界，不自动意味着模型更好或更差。
- 概率用于临床决策前必须在独立数据上检查校准。

## 9. 诊断与稳健性

1. 在嵌套交叉验证中联合调节 $C$ 与 $\gamma$。
2. 比较线性核与 RBF 核，避免无必要的非线性。
3. 报告支持向量比例、类别别敏感度、PR-AUC 和校准。
4. 按患者、中心或时间正确拆分数据。
5. 做外部验证、亚组评估和阈值敏感性分析。

## 10. 推荐可视化

- 二维降维后的边界、间隔和支持向量示意。
- ROC、PR、校准和决策曲线。
- $C$-$\gamma$ 验证性能热图。
- 决策分数按类别的分布。

下图展示降维后三维空间中的 SVM 决策边界：

![](../../04_示例图像/svm_decision_boundary_3d.png)

## 11. 优势、局限与常见坑

**优势：** 高维小样本中常有竞争力，最大间隔有清晰几何意义，核方法灵活。

**局限：** 大样本训练慢，调参敏感，非线性模型难解释，概率需额外处理。

**常见坑：** 不标准化；只看准确率；在测试集选 $C,\gamma$；把决策分数当概率；高维小样本先全数据筛特征再交叉验证。

## 12. 与相近方法的区别

- [[Logistic回归（Logistic Regression）]]：直接建模概率，系数更易解释；SVM 优化间隔。
- [[支持向量回归（Support Vector Regression, SVR）]]：把分类间隔改为连续结局的 $\epsilon$ 管。
- [[随机森林（Random Forest）]]：树通过局部切分形成边界，SVM 通过核相似度形成边界。
- 选择经验：需要概率解释时先考虑 Logistic；高维小样本、判别优先时将线性 SVM 作为强基线。

## 13. 医学研究中的典型应用

- 基因表达、代谢组和蛋白组疾病分类。
- 影像组学良恶性判别与病理亚型预测。
- 临床文本、波形或编码向量分类。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| hyperplane | 线性决策边界 |
| margin | 边界与最近训练点之间的距离 |
| support vector | 决定最优边界的邻界或违例样本 |
| slack variable | 允许进入间隔或错分的程度 |
| hinge loss | 只惩罚函数间隔不足 1 的损失 |

## 15. 相关方法

- [[支持向量回归（Support Vector Regression, SVR）]]
- [[多项式核回归（Polynomial Kernel Regression）]]
- [[Logistic回归（Logistic Regression）]]
- [[随机森林（Random Forest）]]

## 16. 参考资料

- Cortes C, Vapnik V. Support-vector networks. *Mach Learn*. 1995;20:273-297.
- Schölkopf B, Smola AJ. *Learning with Kernels*. MIT Press; 2002.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- scikit-learn Developers. Support Vector Machines User Guide. [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html) （访问日期：2026-07-09）

---
title: 一致性指数
english_name: Concordance Index, C-index
slug: concordance-index-c-index
aliases: [C-index, C统计量, Harrell C, concordance index, "一致性指数（Concordance Index, C-index）"]
category: 模型评估与解释
subcategory: 区分度评估
tags: [医学统计, 数据科学, 区分度, 生存分析, 预测模型]
status: 已建
difficulty: intermediate
question_type: 风险/生存模型的区分度评估
data_type: [表格数据, 生存数据]
outcome_type: [二分类, 时间到事件]
python_packages: [lifelines, scikit-survival]
r_packages: [survival, Hmisc]
---

# 一致性指数（Concordance Index, C-index）

## 1. 方法概览

### 1.1 定义

一致性指数衡量预测模型的区分度：随机取一对可比较个体，模型给「更早发生事件者」更高风险的概率。它是 ROC 曲线下面积在生存数据上的推广。

### 1.2 它主要解决什么问题

- 研究问题：预测模型能否把高风险个体与低风险个体正确排序。
- 适用任务：生存/风险预测模型的区分度评估与比较。
- 常见医学场景：预后评分、复发风险模型、列线图的判别能力评价。

### 1.3 直觉理解

好的风险模型应给「结局更差的人」更高的风险分。C-index 就是把所有能比较的个体两两配对，数一数模型排序正确的比例——0.5 相当于抛硬币，1.0 为完美排序。

## 2. 数学形式

### 2.1 核心公式

$$
C=\frac{\#\{\text{可比对且预测排序一致}\}}{\#\{\text{可比对}\}}
$$

即在所有可比较对 $(i,j)$（结局先后可判定）中，预测风险与实际结局排序一致的比例。

### 2.2 参数或统计量含义

- 可比对：两个体的事件先后可确定（考虑删失后仍可判定者）。
- 一致对：实际先发生事件者被赋予更高风险。
- $C=0.5$ 无区分度，$C=1$ 完美；二分类结局时 $C=\mathrm{AUC}$。
- 时依 C-index 可评估某一时间窗内的区分度。

### 2.3 关键假设

- 风险评分与结局排序的方向一致。
- 删失为非信息性；重删失会使 Harrell C 偏乐观（可用 Uno C 校正）。
- 只衡量排序（区分度），不衡量校准。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：模型输出的风险评分/线性预测子。
- 因变量类型：二分类或时间到事件（含删失）。
- 数据结构：每个体的预测风险 + 实际结局（+ 生存时间与事件指示）。
- 是否适合高维数据：对模型输出评分即可，与维度无关。
- 是否适合缺失较多数据：需完整的评分与结局。
- 是否适合删失数据：适用（Harrell/Uno C）。
- 是否适合重复测量数据：需时依 C-index 扩展。

### 3.2 示例表格

| 个体 | 风险评分 | 生存时间 | 事件 |
| --- | --- | --- | --- |
| 1 | 2.1 | 12 | 1 |
| 2 | 0.4 | 40 | 0 |
| 3 | 1.5 | 20 | 1 |
| 4 | 0.9 | 40 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：预测风险评分、结局（及生存时间/事件）。
- 关键变量：风险评分、time、event。
- 需要预处理的内容：确认评分方向、选择 Harrell 或 Uno C。

#### 产出

- 模型对象/统计结果：C-index 及置信区间。
- 参数估计：一致对比例。
- 预测结果：无（是评估指标）。
- 不确定性指标：C 的 95% CI，两模型 C 的比较检验。

## 4. 适用场景

- 适合：比较模型区分能力、报告预后模型判别度。
- 不适合：评估校准（用 [[校准曲线（Calibration Curve）]]）或临床净获益（用 [[决策曲线分析（Decision Curve Analysis, DCA）]]）。
- 使用前需要特别检查的点：删失比例、是否需 Uno C、是否需时依评估。

## 5. 实现

### 5.1 Python

常用包：

- `lifelines`

```python
from lifelines.utils import concordance_index

# 事件时间、预测风险(越大越危险取负号)、事件指示
c = concordance_index(df["time"], -df["risk_score"], df["event"])
print("C-index =", round(c, 3))
```

### 5.2 R

常用包：

- `survival`

```r
library(survival)
fit <- coxph(Surv(time, event) ~ risk_score, data = df)
summary(fit)$concordance          # C-index 及标准误
# 或直接:
concordance(Surv(time, event) ~ risk_score, data = df)
```

## 6. 结果如何解释

- 核心结果看什么：C 值大小与置信区间是否明显高于 0.5。
- 每个主要参数如何解释：约定俗成——$0.5$ 无用、$0.7$ 尚可、$0.8$ 良好、$>0.9$ 很强（依场景而定）。
- 临床或医学意义如何表达：C 高说明模型能有效排序高低风险，但不代表预测概率准确。
- 常见误读：把高 C 当成校准好；忽视 Harrell C 在重删失下偏乐观。

## 7. 推荐可视化

- 不同模型 C-index 的对比（含 CI）森林图。
- 时依 C-index 曲线。
- 风险分层的 Kaplan-Meier 曲线佐证区分度。

## 8. 优势、局限与常见坑

### 优势

- 单一、可解释的区分度指标，广泛用于预后模型。
- 处理删失，二分类时退化为 AUC。
- 便于跨模型比较。

### 局限

- 只反映排序，不反映校准与临床效用。
- Harrell C 在重删失时偏乐观。
- 对已很强的模型，改进的边际增量常很小。

### 常见坑

- 只报 C-index 而不报校准与净获益。
- 用训练集 C（乐观）而非验证/交叉验证 C。
- 评分方向搞反导致 C<0.5。

## 9. 与相近方法的区别

- 和 [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]] 的区别：AUC 针对固定二分类结局，C-index 推广到含删失的生存结局。
- 和 [[校准曲线（Calibration Curve）]] 的区别：C 看区分度，校准看概率准确性，二者互补。
- 和 [[决策曲线分析（Decision Curve Analysis, DCA）]] 的区别：DCA 评估临床净获益而非排序能力。

## 10. 医学研究中的典型应用

- 肿瘤预后评分/列线图的判别能力评价。
- 心血管风险模型的区分度比较。
- 新生物标志物加入后区分度是否提升。

## 11. 相关方法

- [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[校准曲线（Calibration Curve）]]
- [[决策曲线分析（Decision Curve Analysis, DCA）]]

## 12. 参考资料

- Harrell FE, Lee KL, Mark DB. Multivariable prognostic models. *Stat Med*. 1996;15(4):361-387.
- Uno H, et al. On the C-statistics for evaluating overall adequacy of risk prediction with censored survival data. *Stat Med*. 2011;30(10):1105-1117.
- Pencina MJ, D'Agostino RB. Overall C as a measure of discrimination in survival analysis. *Stat Med*. 2004;23(13):2109-2123.

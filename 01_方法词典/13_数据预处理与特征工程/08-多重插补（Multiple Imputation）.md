---
title: 多重插补
english_name: Multiple Imputation
slug: multiple-imputation
aliases: [MI, multiple imputation, MICE, 链式方程多重插补, "多重插补（Multiple Imputation）"]
category: 数据预处理与特征工程
subcategory: 缺失数据处理
tags: [医学统计, 数据科学, 缺失数据, 插补, 不确定性]
status: 已建
difficulty: intermediate
question_type: 缺失数据处理与不确定性传播
data_type: [表格数据]
outcome_type: [连续型, 二分类, 时间到事件]
python_packages: [statsmodels, scikit-learn]
r_packages: [mice]
---

# 多重插补（Multiple Imputation）

## 1. 方法概览

### 1.1 一句话本质

多重插补不假装缺失值有唯一答案，而是生成多份合理完整数据，分别分析，再把“数据内不确定性”和“不同填法之间的不确定性”一起合并。

### 1.2 定义

多重插补（MI）从缺失值的预测分布生成 $m$ 份不同填补，按同一分析模型得到 $m$ 组估计，最后用 Rubin 规则汇总。MICE/FCS 是常用实现：为每个不完整变量指定条件模型并循环更新。

### 1.3 它主要解决什么问题

- 完整病例分析会浪费样本，并在非 MCAR 时产生偏倚。
- 单次均值/回归插补低估不确定性、扭曲相关结构。
- 医学场景：队列随访、EHR 检验缺失、临床试验协变量和问卷缺失。

### 1.4 直觉与类比

一个缺失肌酐可能是 0.9、1.1 或 1.3，而不是必然等于 1.1。MI 像把这几种合理世界都分析一遍；若结论随填法变化大，最终标准误就应更大。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

缺失值不可观察，任何填补都有额外不确定性。单次插补把预测值当真值，分析模型看不到这层不确定性，置信区间过窄。完整病例则隐含强 MCAR 条件并丢失辅助信息。

### 2.2 关键洞察

把缺失数据问题拆成三个阶段：

1. 从后验预测分布抽取多份缺失值。
2. 在每份数据上使用原本计划的完整数据分析。
3. 用插补间差异补偿额外不确定性。

插补模型应至少包含分析模型中的结局、暴露、协变量，以及与缺失和变量本身相关的辅助变量。

### 2.3 与朴素/相邻做法的对比

- 完整病例简单但可能偏倚、低效。
- 均值插补压缩方差、削弱相关。
- 单次回归插补保留预测关系但仍低估不确定性。
- 最大似然直接在模型内处理缺失；MI 更灵活，可支持多种后续分析。

## 3. 数学形式

### 3.1 核心公式

第 $l$ 份插补数据给出估计 $\hat Q_l$ 和方差 $U_l$。合并点估计：

$$
\bar Q=\frac1m\sum_{l=1}^{m}\hat Q_l
$$

插补内方差：

$$
\bar U=\frac1m\sum_{l=1}^{m}U_l
$$

插补间方差：

$$
B=\frac{1}{m-1}\sum_{l=1}^{m}(\hat Q_l-\bar Q)^2
$$

总方差：

$$
T=\bar U+\left(1+\frac1m\right)B
$$

这个式子在说：最终不确定性等于每份数据本身的分析方差，加上不同合理填法导致的额外波动。

### 3.2 推导脉络

若缺失信息很少，各份估计接近，$B$ 小，MI 接近完整数据分析；若填法影响大，$B$ 增大，标准误自动膨胀。$1+1/m$ 修正有限插补次数带来的 Monte Carlo 误差。

### 3.3 参数与统计量含义

- $m$：插补份数；缺失信息高时应更大，常用 20–100。
- burn-in/iteration：链式方程达到稳定前的迭代。
- $\bar U$：给定填补后的分析不确定性。
- $B$：缺失值未知导致的插补间不确定性。
- $T$：Rubin 总方差。

### 3.4 关键假设（含违反后果）

| 机制/假设 | 含义 | 违反后果/处理 |
| --- | --- | --- |
| MCAR | 缺失与任何数据无关 | 完整病例可无偏但低效 |
| MAR | 给定观测变量后，缺失与缺失值无关 | MI 的常规识别基础 |
| MNAR | 即使控制观测变量，缺失仍依赖未观测值 | 常规 MI 仍偏；做 delta/pattern-mixture 敏感性 |
| 模型相容 | 插补模型不比分析模型更简单 | 交互/非线性被破坏 |
| 聚类结构保留 | 患者/中心相关性进入插补 | 标准误与关联失真 |

## 4. 手把手算例

对同一治疗效应做 $m=3$ 次插补，得到

$$
\hat Q=(1.8,2.0,2.2),\qquad
U=(0.04,0.05,0.06)
$$

**点估计：**

$$
\bar Q=(1.8+2.0+2.2)/3=2.0
$$

**插补内方差：**

$$
\bar U=(0.04+0.05+0.06)/3=0.05
$$

**插补间方差：**

$$
B=\frac{(1.8-2)^2+(2-2)^2+(2.2-2)^2}{2}=0.04
$$

**总方差和标准误：**

$$
T=0.05+\left(1+\frac13\right)0.04=0.1033
$$

$$
SE=\sqrt{0.1033}=0.321
$$

近似 95% 区间为

$$
2.0\pm1.96(0.321)=(1.37,2.63)
$$

若错误地只用 $\bar U$，标准误为 $\sqrt{0.05}=0.224$，明显忽略了不同填法之间的波动。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 多变量表格，连续、二元、有序或计数缺失变量。
- 每类变量应使用匹配条件模型。
- 纵向/中心聚类需多层 MI 或显式聚类变量。
- 生存分析插补应包含事件指示和累计基线风险等信息。

### 5.2 示例表格

| patient_id | treatment | age | creatinine | outcome |
| --- | ---: | ---: | ---: | ---: |
| P01 | 0 | 55 | 1.0 | 8 |
| P02 | 1 | 63 | NA | 6 |
| P03 | 1 | NA | 1.4 | 5 |

### 5.3 输入与产出

输入不完整数据、变量类型、预测矩阵、$m$、迭代数和分析模型；产出 $m$ 份数据、诊断链、合并估计、标准误和敏感性结果。

## 6. 适用场景

- 适合：MAR 在丰富观测变量下较可信，需要保留样本和不确定性。
- 不适合：缺失定义错误、MNAR 风险高却不做敏感性分析、样本极少无法建插补模型。
- 插补设计必须在查看最终结果前制定并完整报告。

## 7. 实现

### 7.1 Python

下面用随机后验抽样生成 5 份插补，并手工按 Rubin 规则合并 OLS 系数。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

df = pd.DataFrame({
    "y": [8, 7, 6, 5, 5, 4],
    "treatment": [0, 0, 1, 1, 1, 0],
    "age": [55, 60, 63, np.nan, 70, 50],
    "creatinine": [1.0, 1.1, np.nan, 1.4, 1.6, 0.9],
})
estimates, variances = [], []
for seed in range(5):
    imp = IterativeImputer(
        sample_posterior=True, random_state=seed, max_iter=20
    )
    complete = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    fit = sm.OLS(
        complete["y"],
        sm.add_constant(complete[["treatment", "age", "creatinine"]]),
    ).fit()
    estimates.append(fit.params["treatment"])
    variances.append(fit.cov_params().loc["treatment", "treatment"])

qbar = np.mean(estimates)
ubar = np.mean(variances)
b = np.var(estimates, ddof=1)
tvar = ubar + (1 + 1 / len(estimates)) * b
print(qbar, np.sqrt(tvar))
```

### 7.2 R

```r
library(mice)

dat <- data.frame(
  y = c(8, 7, 6, 5, 5, 4),
  treatment = c(0, 0, 1, 1, 1, 0),
  age = c(55, 60, 63, NA, 70, 50),
  creatinine = c(1.0, 1.1, NA, 1.4, 1.6, 0.9)
)
set.seed(42)
imp <- mice(dat, m = 20, maxit = 20, printFlag = FALSE)
fit <- with(imp, lm(y ~ treatment + age + creatinine))
summary(pool(fit))
plot(imp)  # 链均值/标准差诊断
```

## 8. 结果如何解读

$\bar Q$ 是跨合理完整数据的平均效应；总标准误同时反映抽样和缺失不确定性。MI 不能“创造信息”，也不证明 MAR；结果应与完整病例和 MNAR 敏感性分析对照。

## 9. 假设诊断与稳健性

- 缺失模式、比例及其与观测变量的关系。
- 链轨迹与分布收敛；比较观测值和插补值分布。
- 检查范围、逻辑约束和派生变量一致性。
- 增加 $m$、迭代数与随机种子，检查 Monte Carlo 稳定性。
- delta adjustment/pattern-mixture 做 MNAR 敏感性。
- 按中心、患者层级和时间顺序保留结构。

## 10. 推荐可视化

- 缺失矩阵/UpSet 图。
- 缺失率与协变量关系图。
- MICE 链轨迹。
- 观测值 vs 插补值密度/条带图。
- 完整病例、MAR MI、MNAR 情景结果森林图。

## 11. 优势、局限与常见坑

### 优势

- 保留样本并传播缺失不确定性。
- 可纳入辅助变量，适配多种后续模型。

### 局限

- 依赖 MAR、插补模型和相容性。
- 多层、非线性和 MNAR 设计较复杂。

### 常见坑

- 只做一次插补。
- 不把结局放入解释变量插补模型。
- 插补后把 $m$ 份数据简单堆叠分析。
- 在交叉验证前全数据插补预测特征。
- 只报告 pooled 结果，不做诊断/MNAR 分析。

## 12. 与相近方法的区别

- 完整病例删除有任一缺失的样本。
- 单次插补没有插补间方差。
- 最大似然在特定模型内直接积分缺失。
- 如何选择：推断任务常用 MI/最大似然；预测任务必须将插补嵌入训练/验证流水线。

## 13. 医学研究中的典型应用

- 队列协变量和实验室指标缺失。
- 临床试验基线/随访数据。
- EHR 信息性检验缺失。
- 生存分析协变量缺失。

需报告每列缺失率、机制假设、变量/方法矩阵、$m$、迭代、诊断、Rubin 合并和 MNAR 敏感性。

## 14. 关键术语

- **MCAR**：缺失与已观测、未观测数据都无关。
- **MAR**：给定已观测数据后，缺失与缺失值无关。
- **MNAR**：缺失仍依赖未观测值。
- **MICE/FCS**：逐变量条件模型循环插补。
- **辅助变量（Auxiliary variable）**：帮助解释缺失或缺失变量的观测变量。
- **Rubin 规则**：合并多份估计及方差的规则。
- **相容性（Congeniality）**：插补模型与分析模型的结构协调。
- **被动插补（Passive imputation）**：由已插补基础变量确定性重算派生变量。

## 15. 相关方法

- 完整病例分析（Complete-case analysis）
- [[广义线性模型（Generalized Linear Model, GLM）]]
- [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[交叉验证（Cross-Validation）]]

## 16. 参考资料

- Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley; 1987.
- van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. CRC Press; 2018.
- Sterne JAC, et al. Multiple imputation for missing data in epidemiological and clinical research. *BMJ*. 2009;338:b2393.
- White IR, Royston P, Wood AM. Multiple imputation using chained equations. *Statistics in Medicine*. 2011;30:377-399.

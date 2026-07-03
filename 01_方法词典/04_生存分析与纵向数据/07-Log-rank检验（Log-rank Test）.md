---
title: Log-rank检验
english_name: Log-rank Test
slug: log-rank-test
aliases: [log-rank test, Mantel-Cox test, "Log-rank检验（Log-rank Test）"]
category: 生存分析与纵向数据
subcategory: 生存曲线比较
tags: [医学统计, 数据科学, 生存分析, 假设检验]
status: 已建
difficulty: basic
question_type: 生存曲线比较
data_type: [生存数据]
outcome_type: [时间到事件]
python_packages: [lifelines]
r_packages: [survival]
---

# Log-rank检验（Log-rank Test）

## 1. 方法概览

### 1.1 一句话本质

Log-rank 检验在每个事件时点摆出一张 2×2 列联表，问「若两组生存真的相同，这组该死几个」，把「实际死的 − 期望死的」沿时间累加——总差距显著偏离 0，就说明两组生存曲线不同。

### 1.2 定义

Log-rank 检验（又称 Mantel-Cox 检验）是比较两组或多组生存曲线是否存在整体差异的非参数检验。它是把每个事件时点的超几何分布信息（观察数 vs 期望数）在时间上累加的分层检验，能自然处理右删失，对**比例风险型**差异最敏感。

### 1.3 它主要解决什么问题

- 研究问题：不同组的生存时间分布是否相同？
- 适用任务：两组或多组 Kaplan-Meier 曲线的显著性比较；随机对照试验中主要生存终点的组间检验。
- 常见医学场景：治疗组 vs 对照组的总生存比较、不同风险分层间的死亡时间比较、不同肿瘤分期的复发时间比较。

### 1.4 直觉与类比

设想两支队伍在同一条时间线上不断有人「出局」。你不能只比总出局人数——因为两组人数、随访长短、删失都不同。公平的裁判方式是：**每当有人出局**，就冻结画面清点「此刻两组各还有几人在场」，据此算「按在场人数比例，这次出局本该落在 A 组几个」。把每一幕的「A 组实际出局 − 期望出局」累加起来：若两组命运相同，正负相消、总和接近 0；若 A 组总是「超额出局」，累加值会明显为正。Log-rank 就是把这份「累积超额」标准化成卡方统计量。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

想比较两组「谁活得久」，朴素做法处处碰壁：比平均生存时间——删失者的时间根本没观测完；比某个固定时点的生存率——选哪个时点主观且浪费其他时点的信息；比总死亡数——不同组人数和随访时长不可比。根本困难是：**如何在删失、组间基础人数不同、且要综合全时段信息的前提下，给出一个公平的组间比较？**

### 2.2 关键洞察

把整个随访切成一串「事件时点」，在每个时点上构造一张 2×2 表（组别 × 生死），**在零假设「两组瞬时风险相同」下，该时点组内死亡数服从超几何分布**——期望和方差都能算出来，且与删失无关（删失只改变下一时点的在场人数）。于是每个时点提供一份「观察 − 期望」及其方差，把它们**跨时点累加**（这正是 Mantel-Haenszel 分层思想），就得到一个综合全时段、正确处理删失的检验统计量。关键在于：**每个时点条件化在「当前在场人数」上，把不可比的原始计数变成了可比的超几何偏差**。

### 2.3 与朴素/相邻做法的对比

- 相对**比较中位生存时间/固定时点生存率**：log-rank 用上全部事件时点，不必主观选时点，检验力通常更高。
- 相对 [[Mantel-Haenszel检验（Mantel-Haenszel Test）]]：log-rank 本质就是「按事件时点分层」的 Mantel-Haenszel，二者同源。
- 相对**加权 log-rank（Wilcoxon-Breslow、Fleming-Harrington）**：标准 log-rank 对各时点等权（对晚期差异更敏感）；加权版可强调早期或晚期差异。
- 相对 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：log-rank 只给 p 值不给效应量；Cox 给出风险比 HR 并能调整协变量。仅一个二分类组时，log-rank 的 score 检验与 Cox 完全一致。

## 3. 数学形式

### 3.1 核心公式

对第 0 组，在每个事件时点 $t_i$ 计算观察减期望，跨时点累加：

$$
U_0=\sum_{i=1}^{D}\bigl(d_{0i}-e_{0i}\bigr),\qquad
e_{0i}=\frac{n_{0i}\,d_i}{n_i},\qquad
v_i=\frac{n_{0i}\,n_{1i}\,d_i\,(n_i-d_i)}{n_i^2\,(n_i-1)}
$$

检验统计量：

$$
\chi^2=\frac{U_0^2}{\sum_i v_i}\ \sim\ \chi^2_{(1)}\quad(\text{两组})
$$

这个式子在说：$e_{0i}$ 是「若两组风险相同、按在场人数比例第 0 组该死的人数」，$U_0$ 是实际与期望之差的累积，$v_i$ 是超几何分布的方差；标准化平方后服从 1 自由度卡方。

### 3.2 推导脉络

1. 在时点 $t_i$，固定「共 $d_i$ 人出事、第 0 组 $n_{0i}$ 人在场、共 $n_i$ 人在场」，则第 0 组死亡数 $d_{0i}$ 服从**超几何分布**，均值 $e_{0i}=n_{0i}d_i/n_i$、方差 $v_i$（上式）。
2. 各时点独立（条件于风险集），故 $U_0=\sum(d_{0i}-e_{0i})$ 均值 0、方差 $\sum v_i$。
3. 由中心极限定理 $U_0/\sqrt{\sum v_i}\approx N(0,1)$，平方即 $\chi^2_{(1)}$。
4. $K$ 组推广：构造 $(K-1)$ 维向量 $\mathbf U$ 与协方差阵 $\mathbf V$，统计量 $\mathbf U^\top\mathbf V^{-}\mathbf U\sim\chi^2_{(K-1)}$。

### 3.3 参数与统计量含义

- $n_{ki}$：时点 $t_i$ 第 $k$ 组在场（at risk）人数。
- $d_{ki}$、$d_i$：时点 $t_i$ 第 $k$ 组、以及合计的事件数。
- $e_{0i}$：零假设下第 0 组的期望事件数。
- $U_0$：观察减期望的累积；符号指示哪组「超额死亡」。
- $O/E$ 比：常报告各组 $\sum d_{ki}/\sum e_{ki}$，直观显示哪组死得比预期多。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 非信息性删失 | 删失与预后无关 | 检验有偏，p 值不可信 | 比较删失者与在随访者特征 |
| 比例风险（隐含最优性） | 组间风险比大致恒定 | 曲线交叉时检验力骤降 | 看 KM 曲线是否交叉 |
| 个体独立 | 事件时间互不影响 | 方差低估、p 值偏小 | 看设计（配对/聚集需分层或分层 log-rank） |
| 事件时点定义清晰 | 时间与事件编码一致 | 列联表构造错误 | 核对 time/event 编码 |

## 4. 手把手算例

沿用生存分析卡的同一份数据，分两组（月，`+` 为删失）：

- **对照组 C**：2, 4, 6+, 8, 10+
- **治疗组 T**：5, 7+, 9, 12+, 14+

对**对照组（第 0 组）**在每个事件时点建表，计算观察减期望：

| 事件时点 $t_i$ | $n_{Ci}$ | $n_{Ti}$ | $n_i$ | 死者在哪组 | $d_i$ | $d_{Ci}$ | 期望 $e_{Ci}=n_{Ci}d_i/n_i$ | $d_{Ci}-e_{Ci}$ | 方差 $v_i$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 5 | 5 | 10 | C | 1 | 1 | 5×1/10 = 0.500 | +0.500 | 0.2500 |
| 4 | 4 | 5 | 9 | C | 1 | 1 | 4×1/9 = 0.444 | +0.556 | 0.2469 |
| 5 | 3 | 5 | 8 | T | 1 | 0 | 3×1/8 = 0.375 | −0.375 | 0.2344 |
| 8 | 2 | 3 | 5 | C | 1 | 1 | 2×1/5 = 0.400 | +0.600 | 0.2400 |
| 9 | 1 | 3 | 4 | T | 1 | 0 | 1×1/4 = 0.250 | −0.250 | 0.1875 |

（删失时点 6+、7+、10+、12+、14+ 不建表，只让下一时点的在场人数相应减少。）

**累加：**

- 对照组观察事件数 $O_C=3$，期望 $E_C=0.500+0.444+0.375+0.400+0.250=1.969$。
- $U_C=O_C-E_C=3-1.969=1.031$。
- $V=\sum v_i=0.2500+0.2469+0.2344+0.2400+0.1875=1.159$。
- 卡方 $\chi^2=U_C^2/V=1.031^2/1.159=\mathbf{0.917}$，df=1。
- 查表 $p\approx 0.34$（$z=U_C/\sqrt V=0.957$）。

**结论：** 对照组实际死了 3 人，而「若两组生存相同」只期望死约 2.0 人——对照组略有超额死亡，方向上治疗组更好，但 $p=0.34\gt 0.05$，差异不显著。这正是小样本（各 5 人、共 5 个事件）的典型结局：即使方向对，证据也不足以下结论。（注：本例数据在 [[Cox比例风险模型（Cox Proportional Hazards Model）]] 卡中拟合出 HR≈0.42，其 score 检验的卡方恰等于这里的 0.917——log-rank 与 Cox 同源的直接印证。）

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：一个分组变量（两类或多类）。
- 因变量类型：时间到事件（time + event）。
- 数据结构：每行一个个体，含 `time`、`event`、`group`。
- 是否适合高维数据：不适合（只比较分组）。
- 是否适合缺失较多数据：time/event 缺失需先处理。
- 是否适合删失数据：适合右删失。
- 是否适合重复测量数据：不适用；聚集数据用分层或稳健版本。

### 5.2 示例表格

Log-rank 直接作用于带分组的生存数据：

| RANDID | TIMEDTH | DEATH | SEX | BMI | AGE_group |
| --- | --- | --- | --- | --- | --- |
| 2448 | 8766 | 0 | 0 | 26.97 | 1 |
| 10552 | 2956 | 1 | 1 | 28.58 | 2 |
| 11252 | 8766 | 0 | 1 | 23.10 | 1 |

按性别分组的典型输出：$\chi^2=78.9$，df=1，$p\lt 2\times10^{-16}$。

### 5.3 输入与产出

#### 输入

- 输入数据：事件时间、事件指示、分组变量。
- 关键变量：`time`、`event`、`group`。
- 需要预处理的内容：删失编码、分组变量清洗、必要时确定分层变量。

#### 产出

- 模型对象/统计结果：卡方统计量、自由度、p 值。
- 参数估计：无回归系数（可报各组 O/E 比作效应方向参考）。
- 预测结果：无。
- 不确定性指标：主要是 p 值；效应量需配 KM 曲线或 Cox 的 HR。

## 6. 适用场景

- 适合：比较两组或多组整体生存差异，尤其风险比大致恒定时。
- 不适合：生存曲线明显交叉（早晚期效应反向）；需要调整混杂协变量；想要效应量。
- 使用前需要特别检查的点：先画 KM 看曲线是否交叉；删失模式；是否需要分层 log-rank（控制中心/分期等）。

## 7. 实现

### 7.1 Python

常用包：

- `lifelines`

```python
from lifelines.statistics import logrank_test, multivariate_logrank_test

# 两组
res = logrank_test(df_A["time"], df_B["time"],
                   event_observed_A=df_A["event"],
                   event_observed_B=df_B["event"])
print(res.test_statistic, res.p_value)

# 多组:一行完成
res_k = multivariate_logrank_test(df["time"], df["group"], df["event"])
res_k.print_summary()
```

### 7.2 R

常用包：

- `survival`

```r
library(survival)

# rho=0 为标准 log-rank; rho=1 为 Peto-Peto(Wilcoxon 型,强调早期)
res <- survdiff(Surv(TIMEDTH, DEATH) ~ sex_label, data = df, rho = 0)
res                       # 输出 N / Observed / Expected / chisq / p

# 分层 log-rank:在 strata() 内控制协变量(如研究中心)
survdiff(Surv(TIMEDTH, DEATH) ~ sex_label + strata(center), data = df)
```

## 8. 结果如何解读

- 核心结果看什么：p 值 + 各组 Observed vs Expected（哪组超额死亡）+ KM 曲线的分离方向。
- 每个主要参数如何解读：$p\lt 0.05$ 说明至少两组生存分布不同；$O_C/E_C\gt 1$ 说明该组死得比预期多（预后差）。
- 临床或医学意义如何表达：必须与 KM 曲线一起呈现，并补充效应量（HR 或某时点生存率差），单独一个 p 值信息量不足。
- 常见误读：显著≠已调整混杂；不显著≠两组相同（可能只是样本小，如本例）；曲线交叉时 p 值大不代表无差异，而是 log-rank 失灵。

## 9. 假设诊断与稳健性

- 曲线交叉：先看 KM。若交叉，标准 log-rank 检验力骤降，改用加权 log-rank（Fleming-Harrington $\rho,\gamma$）或考虑限制平均生存时间（RMST）差异检验。
- 早期/晚期差异：若关心早期差异用 Wilcoxon-Breslow/Peto 加权（给早期事件更大权重）；标准 log-rank 对晚期更敏感。
- 混杂：log-rank 不能调整协变量——用分层 log-rank 控制少数分类混杂，或改用 Cox 回归。
- 聚集/配对数据：独立性被破坏，用分层或稳健方差版本。
- 多重比较：多组两两 log-rank 需校正（如 Bonferroni）。

## 10. 推荐可视化

- 分组 KM 曲线 + 风险集人数表 + log-rank p 值：生存比较的标准配图。
- 各组 O/E 条形或表：直观显示哪组超额死亡。
- 标准 vs 加权 log-rank 对照：当曲线形态复杂时展示结论稳健性。

### 10.1 图像示例

下图给出两组生存曲线对比，是 Log-rank 检验最常见的可视化背景。

![](../../04_示例图像/km_survival_by_sex.png)

## 11. 优势、局限与常见坑

### 优势

- 非参数、标准、实现简单，是生存组间比较的默认检验。
- 正确处理右删失，综合利用全部事件时点。
- 比例风险差异下检验力最优。

### 局限

- 只给 p 值，不给效应量。
- 不能调整协变量。
- 曲线交叉时检验力大幅下降。

### 常见坑

- 只报 p 值不画 KM 曲线。
- 把它当作「任意生存差异」的万能检验（交叉时会漏检）。
- 强混杂场景下不进一步做 Cox。
- 多组两两比较不做多重比较校正。

## 12. 与相近方法的区别

- 和 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：KM 估计并画出曲线，log-rank 检验曲线是否不同——一画一验，配套使用。
- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：单二分类协变量时 log-rank ≈ Cox 的 score 检验；Cox 额外给 HR 和调整能力。需要效应量或调整混杂时用 Cox。
- 和 [[Mantel-Haenszel检验（Mantel-Haenszel Test）]]：log-rank 是按事件时点分层的 Mantel-Haenszel。
- 和加权 log-rank：曲线交叉或关心特定时段时选加权版。
- 如何选择：**先 KM 看形状；不交叉、只需 p 值用标准 log-rank；需效应量或调混杂转 Cox；曲线交叉转加权 log-rank 或 RMST**。

## 13. 医学研究中的典型应用

- 随机对照试验主要生存终点的组间检验（配 KM 主图）。
- 不同风险分层/分期人群时间到事件分布的比较。
- 观察性研究中暴露组与非暴露组生存差异的初步筛查。

## 14. 关键术语

- **超几何分布（Hypergeometric distribution）**：从固定「成功/失败」总体中不放回抽样的分布，log-rank 每个时点期望与方差的来源。
- **观察减期望（Observed − Expected, O−E）**：某组实际事件数减去零假设下期望事件数，log-rank 统计量的核心。
- **风险集（Risk set）**：某时点仍在场（未事件未删失）的个体集合。
- **分层 log-rank（Stratified log-rank）**：在层内计算 O−E 再合并，用于控制分类混杂。
- **加权 log-rank（Weighted log-rank）**：给不同时点事件不同权重（如 Peto-Peto 重早期），应对曲线交叉或特定时段关注。
- **限制平均生存时间（RMST）**：生存曲线在 $[0,\tau]$ 下的面积，其组间差异是曲线交叉时的稳健替代量。

## 15. 相关方法

- [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]
- [[Mantel-Haenszel检验（Mantel-Haenszel Test）]]

## 16. 参考资料

- Peto R, Peto J. Asymptotically efficient rank invariant test procedures. *J R Stat Soc Ser A*. 1972;135(2):185-207.
- Mantel N. Evaluation of survival data and two new rank order statistics arising in its consideration. *Cancer Chemother Rep*. 1966;50(3):163-170.
- Klein JP, Moeschberger ML. *Survival Analysis: Techniques for Censored and Truncated Data*. 2nd ed. Springer; 2003.
- R Core Team / survival package. `survdiff`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survdiff.html](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survdiff.html) （访问日期：2026-07-02）

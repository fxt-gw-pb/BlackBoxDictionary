---
title: 样本量与检验效能
english_name: Sample Size and Power Analysis
slug: sample-size-and-power-analysis
aliases: [power analysis, sample size, 样本量估计, 检验效能, "样本量与检验效能（Sample Size and Power Analysis）"]
category: 研究设计与数据理解
subcategory: 研究设计
tags: [医学统计, 研究设计, 样本量, 检验效能]
status: 已建
difficulty: intermediate
question_type: 研究设计与样本量规划
data_type: [表格数据]
outcome_type: [连续型, 二分类, 时间到事件]
python_packages: [statsmodels]
r_packages: [pwr]
---

# 样本量与检验效能（Sample Size and Power Analysis）

## 1. 方法概览

### 1.1 定义

样本量与检验效能分析在研究开始前，根据预期效应、显著性水平和目标把握度，计算需要多少样本，或评估给定样本量下能检出多大效应。

### 1.2 它主要解决什么问题

- 研究问题：要有把握检出临床上有意义的效应，需要招募多少受试者。
- 适用任务：临床试验/观察研究的样本量估计、事后效能与最小可检测效应评估、方案设计。
- 常见医学场景：RCT 组间比较、诊断/预后研究、生存终点试验的事件数规划。

### 1.3 直觉理解

样本量、效应大小、显著性水平 $\alpha$ 与把握度（power）$1-\beta$ 四者相互牵制：固定其三可解出第四。效应越小、要求把握度越高，所需样本越多。规划阶段做这件事，能避免“做完发现没把握检出真效应”。

## 2. 数学形式

### 2.1 核心公式

把握度是在效应真实存在时拒绝原假设的概率：

$$
\text{power}=1-\beta=P(\text{拒绝}H_0\mid H_1\ \text{为真})
$$

两组均值比较（每组 $n$、双侧 $\alpha$、效应量 $d=\Delta/\sigma$）的近似样本量：

$$
n=\frac{2\,(z_{1-\alpha/2}+z_{1-\beta})^2}{d^2}
$$

生存终点常按所需事件数规划（Schoenfeld）：

$$
D=\frac{4\,(z_{1-\alpha/2}+z_{1-\beta})^2}{(\ln \mathrm{HR})^2}
$$

### 2.2 参数或统计量含义

- $\alpha$：一类错误率（假阳性），常取 0.05。
- $1-\beta$：把握度，常取 0.8 或 0.9。
- 效应量：如 Cohen's $d$、风险差、$\ln\mathrm{HR}$。
- $\sigma$：结局的（预期）标准差；$D$：所需事件数。

### 2.3 关键假设

- 预期效应量、方差、事件率等来自既往研究或临床判断。
- 检验方法、分配比例、单/双侧已预先确定。
- 计入脱落、多重比较、聚类等设计因素后再定最终样本量。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：分组/暴露变量。
- 因变量类型：连续、二分类或时间到事件（各有对应公式）。
- 数据结构：设计阶段的参数假设，而非已有数据。
- 是否适合高维数据：高维/多重检验需相应校正 $\alpha$ 或采用 FDR 视角。
- 是否适合缺失较多数据：预期脱落率应纳入膨胀。
- 是否适合删失数据：生存终点按事件数而非人数规划。
- 是否适合重复测量数据：需用设计效应/混合模型效能。

### 3.2 示例表格

| 结局类型 | 关键输入 | 输出 |
| --- | --- | --- |
| 两组均值 | Δ, σ, α, power | 每组样本量 |
| 两组率 | p1, p2, α, power | 每组样本量 |
| 生存 HR | HR, α, power, 事件率 | 所需事件数与总样本 |

### 3.3 输入与产出

#### 输入

- 输入数据：预期效应、方差/基线率、$\alpha$、目标 power、分配比例。
- 关键变量：效应量、显著性水平、把握度。
- 需要预处理的内容：确定检验类型、单/双侧、脱落与多重性调整。

#### 产出

- 模型对象/统计结果：所需样本量或事件数。
- 参数估计：给定 n 下的把握度、最小可检测效应。
- 预测结果：样本量–把握度曲线。
- 不确定性指标：对效应/方差假设的敏感性分析。

## 4. 适用场景

- 适合：研究方案设计、基金/伦理申请、终点与样本规划。
- 不适合：用“事后把握度”为已获阴性结果辩护（是常见谬误）。
- 使用前需要特别检查的点：效应量来源是否可信、是否计入脱落与多重比较、单双侧设定。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
n = analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8, ratio=1.0)
print("每组所需样本量:", round(n, 1))
```

### 5.2 R

常用包：

- `pwr`

```r
library(pwr)

# 两组均值, Cohen's d = 0.5
pwr.t.test(d = 0.5, sig.level = 0.05, power = 0.8, type = "two.sample")

# 两组率
# power.prop.test(p1 = 0.30, p2 = 0.45, sig.level = 0.05, power = 0.8)
```

## 6. 结果如何解释

- 核心结果看什么：所需样本量/事件数，以及对关键假设的敏感性。
- 每个主要参数如何解释：报告应写明假设的效应量、$\alpha$、power 与来源，便于同行判断合理性。
- 临床或医学意义如何表达：把效应量与“临床最小重要差异”对齐，而非追求统计上最易检出的效应。
- 常见误读：把事后把握度当作阴性结果的解释；用观测效应反算 power（循环论证）。

## 7. 推荐可视化

- 样本量–把握度曲线（不同效应量）。
- 敏感性分析热图（效应量 × 方差）。
- 招募/脱落调整后的样本量条形图。

## 8. 优势、局限与常见坑

### 优势

- 提升研究的可行性与伦理合理性，避免不足或过度招募。
- 迫使研究者事先明确效应量与假设。
- 可比较不同设计的效率。

### 局限

- 高度依赖预期参数，假设错则结论错。
- 不同软件/公式的近似与连续性校正可能有差异。
- 复杂设计（聚类、适应性、多终点）需专门方法。

### 常见坑

- 效应量凭“希望值”而非临床意义设定。
- 忽略脱落率与多重比较膨胀。
- 用事后把握度为阴性结果开脱。

## 9. 与相近方法的区别

- 和具体检验（如 [[两独立样本t检验（Two-Sample t-Test）]]）的关系：效能分析在设计阶段服务于将来要做的检验。
- 和 [[Bootstrap重抽样（Bootstrap Resampling）]] 的关系：复杂设计可用模拟（含 bootstrap/蒙特卡洛）估计把握度。
- 和 [[多重检验与错误率控制（Multiple Testing and Error Rate Control）]] 的关系：多重终点会改变有效 $\alpha$，需在样本量中体现。

## 10. 医学研究中的典型应用

- RCT 主要终点的样本量估计与方案撰写。
- 生存试验按事件数规划随访与入组。
- 诊断/预后研究达到目标区间宽度所需样本。

## 11. 相关方法

- [[两独立样本t检验（Two-Sample t-Test）]]
- [[Log-rank检验（Log-rank Test）]]
- [[多重检验与错误率控制（Multiple Testing and Error Rate Control）]]

## 12. 参考资料

- Chow SC, Shao J, Wang H, Lokhnygina Y. *Sample Size Calculations in Clinical Research*. 3rd ed. Chapman and Hall/CRC; 2017.
- Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. *Biometrics*. 1983;39(2):499-503.
- Champely S. pwr: Basic Functions for Power Analysis. [https://cran.r-project.org/package=pwr](https://cran.r-project.org/package=pwr) （访问日期：2026-07-02）

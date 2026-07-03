---
title: 广义线性混合效应模型
english_name: Generalized Linear Mixed-Effects Model, GLMM
slug: generalized-linear-mixed-effects-model-glmm
aliases: [GLMM, generalized mixed model, "广义线性混合效应模型（Generalized Linear Mixed-Effects Model, GLMM）"]
category: 生存分析与纵向数据
subcategory: 纵向非高斯结局
tags: [医学统计, 数据科学, 纵向数据, 混合效应模型, 二分类]
status: 已建
difficulty: intermediate
question_type: 重复测量二元/计数结局建模
data_type: [纵向数据, 表格数据]
outcome_type: [二分类, 计数型]
python_packages: [statsmodels]
r_packages: [lme4]
---

# 广义线性混合效应模型（Generalized Linear Mixed-Effects Model, GLMM）

## 1. 方法概览

### 1.1 一句话本质

GLMM = [[广义线性模型（Generalized Linear Model, GLM）]] + 随机效应：给每个受试者一条自己的 Logistic/Poisson 曲线（随机截距/斜率），在**连接函数尺度**上围绕总体结构波动，从而对相关的二元/计数重复测量做「同一个体层面」的建模。

### 1.2 定义

广义线性混合效应模型把混合效应思想与 GLM 结合，用于重复测量或层级数据中的二元、计数等**非高斯**结局。它对给定随机效应后的条件分布用指数族 + 连接函数，随机效应服从正态；系数是 **subject-specific（条件）** 效应。

### 1.3 它主要解决什么问题

- 研究问题：结局是二元或计数、且同一主体有重复测量时，如何同时刻画个体差异与个体内相关，并得到「对同一个体」的效应？
- 适用任务：纵向二元结局、重复计数结局、随机截距/随机斜率的非高斯模型。
- 常见医学场景：多次随访是否高血压、是否达标、是否复发；反复计数的发作次数、门诊次数。

### 1.4 直觉与类比

延续 [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]] 的「每人一条线」，但结局现在是「是/否」。想象每个病人有一个自己的「基础患病倾向」——有人天生 log-odds 偏高（易发病），有人偏低。GLMM 给每人一个随机截距去平移他的 Logistic 曲线，再问「在同一个人身上，时间/治疗如何改变他的患病概率」。关键区别在于：因为 Logistic 的 log-odds↔概率是**弯的**（非线性），「对同一个人的效应」和「对整个人群平均的效应」不再相等——这正是 GLMM（条件）与 [[广义估计方程（Generalized Estimating Equations, GEE）]]（边际）分道扬镳之处。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

LMM 靠正态假设，边际与条件恰好一致，日子好过。但二元结局既非正态、方差又随均值变（$\text{Var}=p(1-p)$），且重复测量相关。若直接对二元纵向数据套普通 Logistic，独立性假设被破坏、SE 全错。若照搬 LMM 那套「加个正态随机截距」，又会撞上一个新麻烦：**连接函数是非线性的**，随机效应在 log-odds 尺度上对称，映射回概率尺度就不对称了。根本困难是：**如何对相关的非高斯结局建模，同时厘清『效应到底是对个体还是对人群』？**

### 2.2 关键洞察

在**连接函数（线性预测子）尺度**上加正态随机效应，让条件独立性在给定 $\mathbf b_i$ 后成立：

$$
g(\mu_{ij})=\mathbf X_{ij}^\top\boldsymbol\beta+\mathbf Z_{ij}^\top\mathbf b_i
$$

于是同一人重复测量的相关，全部由共享的 $\mathbf b_i$ 承担。代价与洞察并存：由于 $g^{-1}$ 非线性，对随机效应求期望（积分掉 $\mathbf b_i$）得到的**边际均值不再有简洁形式**，且**条件系数 $\boldsymbol\beta$ 的绝对值大于对应的边际系数**——随机效应方差越大，两者差得越远。理解这一点，就理解了 GLMM 报告的是「保持这个人的随机效应不变时」的效应。

### 2.3 与朴素/相邻做法的对比

- 相对 [[Logistic回归（Logistic Regression）]] / [[Poisson回归（Poisson Regression）]]：GLMM 显式处理个体内相关，SE 才对；系数解释从「人群」变成「个体」。
- 相对 [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]：LMM 是 GLMM 在「正态 + 恒等连接」下的特例，那时条件=边际。
- 相对 [[广义估计方程（Generalized Estimating Equations, GEE）]]：**同一份二元纵向数据**，GLMM 给条件（subject-specific）OR、GEE 给边际（population-average）OR，前者绝对值更大。要「对病人个体」的效应用 GLMM，要「对人群平均」的效应用 GEE。
- 拟合难度：GLMM 的似然含对随机效应的积分，需数值方法（拉普拉斯近似、自适应高斯-埃尔米特求积、或贝叶斯），比 GEE 重。

## 3. 数学形式

### 3.1 核心公式

$$
\begin{aligned}
Y_{ij}\mid\mathbf b_i &\sim \text{指数族（如 Bernoulli、Poisson）} \\
g(\mu_{ij}) &= \eta_{ij}=\mathbf X_{ij}^\top\boldsymbol\beta+\mathbf Z_{ij}^\top\mathbf b_i \\
\mathbf b_i &\sim N(\mathbf 0,\mathbf D)
\end{aligned}
$$

这个式子在说：给定这个人的随机效应 $\mathbf b_i$ 后，结局服从相应指数族分布；其均值经连接函数 $g$ 连到「总体固定部分 + 个体随机部分」的线性预测子上。二元结局取 logit 连接时，$\boldsymbol\beta$ 就是 subject-specific 的 log-OR。

### 3.2 推导脉络

1. 选分布族（Bernoulli/Poisson）与连接（logit/log），条件均值 $\mu_{ij}=g^{-1}(\eta_{ij})$。
2. 边际似然需把随机效应积掉：$L(\boldsymbol\beta,\mathbf D)=\prod_i\int\prod_j f(y_{ij}\mid\mathbf b_i)\,\phi(\mathbf b_i;\mathbf D)\,d\mathbf b_i$——**无闭式解**，用拉普拉斯近似或求积法数值逼近。
3. 边际-条件关系（logit + 随机截距 $b_i\sim N(0,\tau^2)$）：近似有

$$
\boldsymbol\beta_{\text{边际}}\approx\frac{\boldsymbol\beta_{\text{条件}}}{\sqrt{1+0.346\,\tau^2}}
$$

即条件系数被「衰减因子」缩小后才是边际系数。$\tau^2$ 越大，衰减越强。

### 3.3 参数与统计量含义

- 固定效应 $\boldsymbol\beta$：**subject-specific** 效应——在给定同一个体随机效应时，协变量对 log-odds/log-count 的作用。
- 随机效应 $\mathbf b_i$：主体专属偏移；随机截距方差 $\tau^2$ 越大，个体异质性越强。
- 边际均值 $E(Y_{ij})$：一般无简洁闭式（需对 $\mathbf b_i$ 积分）。
- 拟合方法：拉普拉斯近似（默认）、自适应求积（nAGQ\>1 更准）。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 分布族/连接匹配结局 | Bernoulli+logit、Poisson+log 等 | 系数与推断有偏 | 结局类型核对、残差诊断 |
| 随机效应正态 | $\mathbf b_i\sim N(0,\mathbf D)$ | 估计有偏（比 LMM 更敏感） | 随机效应分布图 |
| 给定 $\mathbf b_i$ 后条件独立 | 相关全由随机效应承担 | 剩余相关未建模 | 残余相关检查 |
| 数值近似足够精 | 拉普拉斯/求积收敛 | 方差成分偏（尤其少测量二元） | 提高 nAGQ 看是否变动 |
| 无过度离散（计数） | Poisson 方差=均值 | SE 偏小 | 加观测级随机效应或用负二项 |

## 4. 手把手算例

GLMM 的似然要对随机效应做积分，**没法像 Logistic 那样几行手推到收敛**——这是它的固有难点，诚实说明并不逐项手算参数。改用一个能手算的**条件-边际换算**例子，把 GLMM 的招牌特征「条件效应比边际大」量化出来（数据与 [[广义估计方程（Generalized Estimating Equations, GEE）]] 卡共用）。

**设定：** 10 名受试者，两期随访，结局为「是否高血压」。观测到的**人群比例**：期 1 有 3/10=30% 高血压，期 2 有 6/10=60%。个体间存在异质性，设随机截距方差 $\tau^2=2$。

**Step 1：先算边际（人群平均）时间效应**——这是 GEE 会给的、也是能从比例直接手算的量。

$$
\text{logit}(0.30)=\log\frac{0.30}{0.70}=-0.847,\qquad
\text{logit}(0.60)=\log\frac{0.60}{0.40}=+0.405
$$

$$
\beta_{\text{边际}}=0.405-(-0.847)=1.253,\qquad OR_{\text{边际}}=e^{1.253}=3.50
$$

**Step 2：换算成 GLMM 的条件（subject-specific）效应。** 由 §3.2 的关系反解：

$$
\beta_{\text{条件}}=\beta_{\text{边际}}\times\sqrt{1+0.346\,\tau^2}
=1.253\times\sqrt{1+0.346\times2}
=1.253\times1.301=\mathbf{1.63}
$$

$$
OR_{\text{条件}}=e^{1.63}=\mathbf{5.10}
$$

**Step 3：读两个数的差别。**

- **GLMM（条件）OR = 5.10**：读作「**对同一个病人**而言，从期 1 到期 2，其患高血压的 odds 增到约 5.1 倍」。
- **GEE（边际）OR = 3.50**：读作「**整个人群**中，期 2 相对期 1 的患病 odds 是 3.5 倍」。
- 同一份数据，条件效应（5.10）明显大于边际效应（3.50）——因为 logit 是弯的，个体异质性 $\tau^2$ 把人群平均曲线「压平」了。若 $\tau^2=0$（人人相同），衰减因子为 1，两者相等。

**结论：** GLMM 报的是 subject-specific 效应，天然比 GEE 的 population-average 效应绝对值大，差距随个体异质性 $\tau^2$ 增大。**两个数都对，只是回答不同的问题**——「这个治疗对你个人有多大用」（条件）vs「在全人群铺开有多大用」（边际）。汇报时务必说明是哪一种，否则读者会把 5.10 和 3.50 当成矛盾。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：时间/期别、处理组、性别、年龄等。
- 因变量类型：二分类、计数型（及其他指数族）。
- 数据结构：long format 重复测量数据，需个体 ID。
- 是否适合高维数据：非默认首选。
- 是否适合缺失较多数据：MAR 下用似然，比传统法灵活。
- 是否适合删失数据：不适合删失结局本身。
- 是否适合重复测量数据：主场。

### 5.2 示例表格

`Framingham_data.csv` 中把 `PREVHYP`（是否高血压）当二元重复结局：

| RANDID | PERIOD | SEX | BMI | PREVHYP |
| --- | --- | --- | --- | --- |
| 6238 | 1 | 1 | 28.73 | 0 |
| 6238 | 2 | 1 | 29.43 | 0 |
| 6238 | 3 | 1 | 28.50 | 0 |
| 11263 | 1 | 1 | 30.30 | 1 |
| 11263 | 2 | 1 | 31.36 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：long format 二元/计数结局数据。
- 关键变量：主体 ID、时间、结局、协变量、随机效应结构。
- 需要预处理的内容：长表整理、结局编码、缺失处理、时间形式。

#### 产出

- 模型对象/统计结果：固定效应（条件系数）、随机效应方差、近似对数似然、AIC。
- 参数估计：subject-specific 效应与 $\tau^2$。
- 预测结果：给定随机效应的条件概率/计数、个体曲线。
- 不确定性指标：固定效应 SE 与区间。

## 6. 适用场景

- 适合：重复测量二元/计数结局，且关心**个体层面**效应或个体预测。
- 不适合：只要人群平均效应（GEE 更省事、更稳健）；结局连续正态（用 LMM）。
- 使用前需要特别检查的点：随机效应结构、数值近似（nAGQ）、收敛、每主体测量数是否够（二元 + 单次测量几乎无信息）。

## 7. 实现

### 7.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

# 频率派 GLMM(二元, 随机截距): PenalizedQMLE
fit = smf.mixedlm  # 注: statsmodels 的二元 GLMM 支持有限
# 推荐用 BinomialBayesMixedGLM(变分贝叶斯)或直接上 R/lme4
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
model = BinomialBayesMixedGLM.from_formula(
    "PREVHYP ~ PERIOD + SEX", {"RANDID": "0 + C(RANDID)"}, data=df)
result = model.fit_vb()
print(result.summary())        # 固定效应为 subject-specific 系数
```

### 7.2 R

常用包：

- `lme4`

```r
library(lme4)

# 二元结局 + 随机截距; nAGQ 提高数值精度
fit <- glmer(PREVHYP ~ PERIOD + SEX + (1 | RANDID),
             family = binomial(link = "logit"),
             data = df, nAGQ = 10)
summary(fit)                       # Estimate 为条件(subject-specific)log-OR
exp(fixef(fit))                    # 条件 OR
VarCorr(fit)                       # 随机截距方差 tau^2
```

## 8. 结果如何解读

- 核心结果看什么：固定效应（**条件** log-OR/OR）方向大小、随机效应方差 $\tau^2$、收敛与近似诊断。
- 每个主要参数如何解读：`PERIOD` 的 OR=5.10 读作「对同一个体，每过一期患病 odds 增到 5.1 倍」；$\tau^2$ 大说明个体基础风险差异大。
- 临床或医学意义如何表达：GLMM 适合回答「对某位病人，干预会如何改变其自身概率」。
- 常见误读：把 GLMM 条件系数当人群平均效应（它比 GEE 边际系数大）；忽视收敛警告仍解读结果；二元 + 每人测量太少时高估随机效应方差。

## 9. 假设诊断与稳健性

- 数值近似：把拉普拉斯（nAGQ=1）换成自适应求积（nAGQ=10）重拟合，若系数/方差明显变动说明近似不足。
- 随机效应分布：画随机截距 BLUP 分布查正态与离群个体。
- 过度离散（计数）：Poisson GLMM 若残差离散大，加**观测级随机效应**或改负二项 GLMM。
- 收敛：留意 `singular fit`/不收敛——常因随机结构过复杂或数据太稀疏，简化随机效应或增数据。
- 稳健性：主体数少、每主体测量少时方差成分不稳；可与 GEE 结果交叉印证。

## 10. 推荐可视化

- 个体二元轨迹图（每人的 0/1 序列）：展示个体异质性。
- 个体拟合概率曲线叠加人群平均：直观区分条件与边际。
- 随机截距分布/caterpillar 图：展示个体基础风险差异。

### 10.1 图像示例

下图展示部分受试者的高血压状态随时间变化轨迹，突出 GLMM 所关注的个体特异性结构。

![](../../04_示例图像/glmm_prehyp_subject_paths.png)

## 11. 优势、局限与常见坑

### 优势

- 同时处理相关数据与非高斯结局。
- 显式建模个体异质性，给 subject-specific 效应与个体预测。
- 随机效应结构灵活（截距/斜率、多层嵌套）。

### 局限

- 似然含积分，拟合计算重、可能不收敛。
- 不同近似方法结果略有差异。
- 系数解释偏条件化，易与边际混淆。

### 常见坑

- 把条件系数直接当人群平均效应汇报。
- 忽视收敛/奇异拟合警告。
- 数据稀疏却设复杂随机结构。
- Poisson GLMM 忽略过度离散。

## 12. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] / [[Poisson回归（Poisson Regression）]]：GLMM 加随机效应处理相关，系数从人群变个体。
- 和 [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]：LMM 是 GLMM 在正态+恒等连接下的特例（那时条件=边际）。
- 和 [[广义估计方程（Generalized Estimating Equations, GEE）]]：GLMM 条件、GEE 边际；同数据 GLMM 系数绝对值更大（差距随 $\tau^2$）。想要个体效应用 GLMM，想要人群效应用 GEE。
- 如何选择：**非高斯结局 + 要个体层面效应/个体预测 → GLMM；只要人群平均、图稳健省心 → GEE；连续正态 → LMM**。

## 13. 医学研究中的典型应用

- 多次随访高血压/达标状态的个体层面建模。
- 反复记录的二元症状或计数事件（发作次数、就诊次数）分析。
- 多中心试验中中心随机效应 + 二元结局。

## 14. 关键术语

- **subject-specific（条件）效应**：在给定同一个体随机效应时的效应，GLMM 的系数含义。
- **population-average（边际）效应**：对整个人群平均的效应，GEE 的系数含义；非线性连接下小于条件效应。
- **连接函数（Link function）**：把均值映到线性预测子的函数（logit、log 等）。
- **随机截距方差 $\tau^2$**：个体基础水平差异的大小；越大条件与边际差越远。
- **拉普拉斯近似 / 自适应求积（nAGQ）**：GLMM 边际似然中随机效应积分的数值逼近方法。
- **观测级随机效应（OLRE）**：给每次观测加随机效应以吸收过度离散。
- **衰减因子（Attenuation factor）**：$\sqrt{1+0.346\tau^2}$，条件系数除以它约等于边际系数。

## 15. 相关方法

- [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]]
- [[Logistic回归（Logistic Regression）]]
- [[广义估计方程（Generalized Estimating Equations, GEE）]]
- [[广义线性模型（Generalized Linear Model, GLM）]]

## 16. 参考资料

- Breslow NE, Clayton DG. Approximate inference in generalized linear mixed models. *J Am Stat Assoc*. 1993;88(421):9-25.
- Molenberghs G, Verbeke G. *Models for Discrete Longitudinal Data*. Springer; 2005.
- Fitzmaurice GM, Laird NM, Ware JH. *Applied Longitudinal Analysis*. 2nd ed. Wiley; 2011.
- CRAN. Package `lme4`. [https://cran.r-project.org/web/packages/lme4/index.html](https://cran.r-project.org/web/packages/lme4/index.html) （访问日期：2026-07-02）

# Framingham Heart Study — BMI 与心血管结局的统计分析

一个基于 Framingham 心脏研究数据的端到端数据分析项目。项目以一条主线问题为核心，从描述性探索逐步深入到回归、纵向与生存建模，系统刻画个体 **BMI** 与**血清总胆固醇**及**生存结局**之间的关系。

## 研究问题

主线问题：理解个体 **BMI** 与其**血清总胆固醇**水平之间的关系，并校正、考察**年龄组**与**性别**的作用与效应修饰；进一步将结局拓展到**死亡/生存**，考察 BMI 与生存的关联。胆固醇水平既可作为连续变量（常做对数变换），也可作为二分类变量（总胆固醇 ≥200 mg/dL 为 "undesirable"，否则 "desirable"）。

## 数据

原始数据统一存放于 [`rawdata/`](rawdata/)，含 4,215 名受试者、三期检查共 11,168 条观测。变量字典与数据背景见 [`rawdata/README.md`](rawdata/README.md)。

## 分析流程

项目按方法分为 7 个模块，由浅入深构成一条完整的分析管线。每个模块目录下的 `README.md` 给出该模块的研究问题、数据范围与方法。

| 模块 | 主题 | 关注点 | 数据范围 |
|------|------|--------|---------|
| [`01-Data Visualization/`](01-Data%20Visualization/) | 数据可视化 | 结局与协变量的分布及边际关系 | 基线 PERIOD 1 |
| [`02-Nonparametric Statistics/`](02-Nonparametric%20Statistics/) | 非参数统计 | 正态性、变换、相关与 bootstrap 标准误 | 基线 |
| [`03-Categorical Data Analysis/`](03-Categorical%20Data%20Analysis/) | 分类数据分析 | 胆固醇状态与 BMI/年龄/性别的独立性 | 基线 |
| [`04-Linear Regression/`](04-Linear%20Regression/) | 线性回归与 ANOVA | 对数胆固醇对 BMI/年龄组/性别的回归与诊断 | 基线 |
| [`05-Generalized Linear Models/`](05-Generalized%20Linear%20Models/) | 广义线性模型 | 胆固醇状态的 logistic 回归与 deviance | 基线 |
| [`06-Longitudinal Data Analysis/`](06-Longitudinal%20Data%20Analysis/) | 纵向数据分析 | 时变 BMI 的 LMM / GLMM / GEE | 全部三期 |
| [`07-Survival Analysis/`](07-Survival%20Analysis/) | 生存分析 | BMI 与生存的 K–M 曲线与 Cox 回归 | 基线 PERIOD 1 |

## 每个模块包含什么

每个模块目录是一份**面向初学者的可复现分析**，统一包含：

| 文件 | 作用 |
|------|------|
| `README.md` | 该模块的**研究问题**（一页纸的问题描述，分析入口）|
| `report.md` | **学习笔记 / 分析报告**：分小节的 R 代码 + 运行结果 + 图 + 逐段讲解，**面向生物统计初学者** |
| `report.Rmd` | 上述报告的源文件（R Markdown）|
| `code.R` | 从 Rmd 抽取的**纯 R 代码**，可直接运行 |
| `figures/` | 报告中的所有图（自动生成）|

> 👉 想学某个主题，直接打开该模块的 **`report.md`**：每段代码后面紧跟它的真实输出与中文解释。

## 复现方式

报告由 **R 4.5.2 + knitr** 生成，代码与结果保证一致。在任一模块目录下运行：

```r
# 重新跑通代码并生成报告（report.md）与图（figures/）
knitr::knit("report.Rmd", "report.md")
# 或只导出纯代码
knitr::purl("report.Rmd", "code.R")
```

主要依赖包：`tidyverse`、`patchwork`、`boot`、`nortest`、`car`、`broom`、`lme4`、`geepack`、`survival`。

## 目录结构

```
framingham research/
├── README.md                 项目总览（本文件）
├── rawdata/                  原始数据与变量字典
│   ├── Framingham_data.csv
│   └── README.md
├── 01-Data Visualization/    README.md · report.md · report.Rmd · code.R · figures/
├── 02-Nonparametric Statistics/
├── 03-Categorical Data Analysis/
├── 04-Linear Regression/
├── 05-Generalized Linear Models/
├── 06-Longitudinal Data Analysis/
└── 07-Survival Analysis/
```

## 一条贯穿全程的发现链

各模块不是孤立的练习，而是层层递进、相互印证的同一个故事：

1. **可视化**（01）：BMI↑、年龄↑ → 胆固醇更高；图上看年龄似乎改变 BMI 的斜率。
2. **非参数**（02）：BMI 与胆固醇是**弱正相关**（Spearman ≈ 0.15），bootstrap 与公式标准误互证。
3. **分类数据**（03）：年龄、BMI 与"胆固醇不理想"显著相关；但**性别在 200 二分切点下不显著**——二分化损失信息。
4. **线性回归**（04）：连续结局下性别**重新变显著**；交互检验证实 **年龄修饰 BMI 效应、性别不修饰**。
5. **GLM**（05）：deviance 的差值不随数据形式改变；年龄的线性趋势有**轻微 lack-of-fit**。
6. **纵向**（06）：胆固醇随时间呈**二次（先升后降）**；个体内相关很强（ICC≈0.69）；GLMM 与 GEE 的系数差异源自**个体特异 vs 总体平均**。
7. **生存**（07）：BMI 与死亡呈 **U 形**（风险最低 BMI≈27）；高血压的危害**随年龄递减**；比例风险假设对 BMI 成立。

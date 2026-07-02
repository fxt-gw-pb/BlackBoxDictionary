# 05 · 广义线性模型（Generalized Linear Models）

> 以二分类胆固醇状态为结局，建立对年龄组的 logistic 回归，并辨析分组与非分组数据下 deviance 的行为。

📘 **完整分析报告 / 学习笔记**：[report.md](report.md)（源码 [report.Rmd](report.Rmd)，纯代码 [code.R](code.R)）

- **数据**：[`../rawdata/Framingham_data.csv`](../rawdata/Framingham_data.csv)，基线
- **方法**：logistic 回归、deviance 与似然比检验、分组/非分组数据

## 研究问题

为**二分类胆固醇变量** $Y$（"undesirable"：总胆固醇 ≥200 mg/dL；否则 "desirable"）建立对**年龄组变量** $X$（取分值 1、2、3）的回归模型，使用基线数据。

记两个模型：

- $M_0$：$\operatorname{logit}[P(Y=1)] = \alpha$
- $M_1$：$\operatorname{logit}[P(Y=1)] = \alpha + \beta X$

以**两种方式**构造数据集：**非分组（ungrouped）**与**分组（grouped）**。

**(a)** 证明 $M_0$ 与 $M_1$ 的偏差（deviance）在两种数据录入形式下**不同**。为什么会这样？

**(b)** 证明 $M_0$ 与 $M_1$ 偏差之**差**在两种数据录入形式下**相同**。为什么？（因此，对于检验 $X$ 的效应，数据以何种形式录入并不影响结果。）

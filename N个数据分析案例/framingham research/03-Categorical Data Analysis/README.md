# 03 · 分类数据分析（Categorical Data Analysis）

> 以二分类胆固醇状态为结局，检验其与 BMI、年龄组、性别的独立性。

📘 **完整分析报告 / 学习笔记**：[report.md](report.md)（源码 [report.Rmd](report.Rmd)，纯代码 [code.R](code.R)）

- **数据**：[`../rawdata/Framingham_data.csv`](../rawdata/Framingham_data.csv)，基线
- **方法**：列联表、卡方/Fisher 独立性检验、趋势检验

## 研究问题

以**二分类胆固醇状态**为结局（"undesirable"：总胆固醇 ≥200 mg/dL；否则 "desirable"）。在**基线**分别检验它与 **BMI、年龄组、性别**的独立性。对每一项检验，按以下步骤进行：

**(a) 可视化**：用图形或表格展示数据。

**(b) 方法与 p 值**：说明所选用的检验方法，并报告 p 值。

**(c) 解释**：对结果做出解释。

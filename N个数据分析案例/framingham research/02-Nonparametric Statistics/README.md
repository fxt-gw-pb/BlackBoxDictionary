# 02 · 非参数统计（Nonparametric Statistics）

> 评估 BMI 与胆固醇的分布形态，选择稳健的相关度量，并用 bootstrap 量化不确定性。

📘 **完整分析报告 / 学习笔记**：[report.md](report.md)（源码 [report.Rmd](report.Rmd)，纯代码 [code.R](code.R)）

- **数据**：[`../rawdata/Framingham_data.csv`](../rawdata/Framingham_data.csv)，基线 BMI 与胆固醇
- **方法**：正态性检验、变量变换、秩相关、bootstrap 标准误

## 研究问题

考虑 Framingham 数据中**基线（baseline）**的 BMI 与胆固醇数据。

**(a) BMI 的正态性。** 检验 BMI 数据的正态性。若偏离正态，应如何对 BMI 数据做变换？变换后再次检验正态性。借助数据可视化对这些结果加以解释。

**(b) 胆固醇的正态性。** 对胆固醇数据重复上述步骤。

**(c) 相关性度量。** 实际中为便于解释，通常更倾向于使用未变换的数据。为度量 BMI 与胆固醇水平之间的关系，选择一种合适的相关方法；报告相关系数，用解析公式计算其标准误，并报告 p 值以评估相关性的显著性。

**(d) Bootstrap 标准误。** 用 bootstrap 估计 (c) 中相关系数的标准误。画出 bootstrap 相关系数重抽样的直方图。其抽样分布看起来是否服从正态？计算 bootstrap 标准误，并与 (c) 中的解析标准误进行比较。

# 06 · 纵向数据分析（Longitudinal Data Analysis）

> 利用三期重复测量，建模时变 BMI 对胆固醇的总体平均效应与个体特异效应。

📘 **完整分析报告 / 学习笔记**：[report.md](report.md)（源码 [report.Rmd](report.Rmd)，纯代码 [code.R](code.R)）

- **数据**：[`../rawdata/Framingham_data.csv`](../rawdata/Framingham_data.csv)，**全部三期**纵向数据
- **方法**：个体轨迹图、线性混合模型（LMM）、广义线性混合模型（GLMM）、广义估计方程（GEE）

## 研究问题

研究个体的 **BMI（作为时变变量）**与其**胆固醇水平**之间的关系，使用全部三次检查的纵向数据。胆固醇水平既可作为连续变量（对数变换），也可作为二分类变量（"undesirable" vs. "desirable"，以 "desirable" 为参照）。

模型应恰当刻画胆固醇水平随时间（`PERIOD` 变量）的变化模式，同时校正**年龄组**（作为分类变量）与**性别**。BMI 对胆固醇状态的**总体平均效应（population-average）**与**个体特异效应（individual-specific）**均为关注对象。具体步骤如下：

**(a) 个体轨迹。** 画出（对数变换后）连续胆固醇水平的个体轨迹。胆固醇随时间的变化模式是线性还是二次？在后续模型中据此建模观察到的模式。

**(b) LMM。** 为（对数变换后）连续胆固醇水平拟合线性混合模型（LMM）。提示：纳入协变量 BMI、period、年龄组与性别。

**(c) GLMM。** 为二分类胆固醇状态拟合广义线性混合模型（GLMM）。

**(d) GEE。** 为二分类胆固醇状态拟合广义估计方程（GEE）。

**(e) 比较。** 比较 GLMM 与 GEE 给出的 BMI 与胆固醇状态关系的结果，并解释二者差异。

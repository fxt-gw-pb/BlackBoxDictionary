## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7.5, fig.height = 4.6,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)   # 含 dplyr / ggplot2 / tidyr 等，最主流的数据处理 + 绘图工具
library(patchwork)   # 把多张 ggplot 拼到一起

# 自定义统一主题，让所有图风格一致
theme_set(theme_minimal(base_size = 12))

# 读入原始数据（项目共享的 rawdata）
raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)

# 整理成"基线分析数据"
df <- raw %>%
  filter(PERIOD == 1) %>%                       # 只保留第 1 期（基线）
  mutate(
    chol_status = factor(if_else(TOTCHOL >= 200, "Undesirable", "Desirable"),
                         levels = c("Desirable", "Undesirable")),
    AGE_group   = factor(AGE_group, levels = c(1, 2, 3),
                         labels = c("≤55", "55–65", ">65")),
    Sex         = factor(SEX, levels = c(0, 1), labels = c("Men", "Women"))
  )

dim(df)                                          # 基线样本量 = 行数 × 列数
colSums(is.na(df[c("TOTCHOL", "BMI", "AGE_group", "SEX")]))  # 各关键变量缺失个数


## ----summary------------------------------------------------------------------
summary(df$TOTCHOL)   # 连续结局的五数概括
summary(df$BMI)       # 连续协变量
table(df$chol_status) # 二分类结局
table(df$AGE_group)   # 年龄组人数
table(df$Sex)         # 性别人数


## ----dist-totchol, fig.height=4-----------------------------------------------
p_hist <- ggplot(df, aes(TOTCHOL)) +
  geom_histogram(binwidth = 10, fill = "#4C72B0", colour = "white") +
  labs(title = "总胆固醇分布（基线）", x = "总胆固醇 (mg/dL)", y = "人数")

p_dens <- ggplot(df, aes(TOTCHOL)) +
  geom_density(fill = "#4C72B0", alpha = 0.5) +
  geom_vline(xintercept = 200, linetype = 2, colour = "firebrick") +
  labs(title = "总胆固醇密度", x = "总胆固醇 (mg/dL)", y = "密度")

p_hist | p_dens     # patchwork：用 | 左右拼图


## ----dist-cholstatus, fig.height=4--------------------------------------------
ggplot(df, aes(chol_status, fill = chol_status)) +
  geom_bar(width = 0.65, show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
  labs(title = "胆固醇状态分布（基线）", x = NULL, y = "人数")


## ----dist-bmi, fig.height=4---------------------------------------------------
ggplot(df, aes(BMI)) +
  geom_histogram(binwidth = 1, fill = "#55A868", colour = "white") +
  labs(title = "BMI 分布（基线）", x = "BMI (kg/m²)", y = "人数")


## ----dist-cat, fig.height=4---------------------------------------------------
p_age <- ggplot(df, aes(AGE_group, fill = AGE_group)) +
  geom_bar(width = 0.65, show.legend = FALSE) +
  labs(title = "年龄组分布", x = "年龄组", y = "人数")

p_sex <- ggplot(df, aes(Sex, fill = Sex)) +
  geom_bar(width = 0.65, show.legend = FALSE) +
  labs(title = "性别分布", x = NULL, y = "人数")

p_age | p_sex


## ----rel-bmi------------------------------------------------------------------
ggplot(df, aes(BMI, TOTCHOL)) +
  geom_point(alpha = 0.25, colour = "#4C72B0") +
  geom_smooth(method = "loess", colour = "firebrick", se = TRUE) +
  labs(title = "总胆固醇 vs BMI", x = "BMI (kg/m²)", y = "总胆固醇 (mg/dL)")


## ----rel-box, fig.height=4----------------------------------------------------
p_a <- ggplot(df, aes(AGE_group, TOTCHOL, fill = AGE_group)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  labs(title = "胆固醇 by 年龄组", x = "年龄组", y = "总胆固醇 (mg/dL)")

p_s <- ggplot(df, aes(Sex, TOTCHOL, fill = Sex)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  labs(title = "胆固醇 by 性别", x = NULL, y = "总胆固醇 (mg/dL)")

p_a | p_s


## ----rel-prop, fig.height=4---------------------------------------------------
p_age_prop <- ggplot(df, aes(AGE_group, fill = chol_status)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "胆固醇状态 by 年龄组", x = "年龄组", y = "比例", fill = "状态")

p_sex_prop <- ggplot(df, aes(Sex, fill = chol_status)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "胆固醇状态 by 性别", x = NULL, y = "比例", fill = "状态")

p_age_prop | p_sex_prop


## ----rel-bmi-box, fig.height=4------------------------------------------------
ggplot(df, aes(chol_status, BMI, fill = chol_status)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  labs(title = "BMI by 胆固醇状态", x = "胆固醇状态", y = "BMI (kg/m²)")


## ----effmod, fig.width=9, fig.height=4.2--------------------------------------
ggplot(df, aes(BMI, TOTCHOL, colour = Sex)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ AGE_group) +
  scale_colour_manual(values = c("Men" = "#4C72B0", "Women" = "#C44E52")) +
  labs(title = "BMI 与总胆固醇的关系：按性别上色、按年龄组分面",
       x = "BMI (kg/m²)", y = "总胆固醇 (mg/dL)", colour = "性别")


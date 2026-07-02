## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7, fig.height = 4.2,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
theme_set(theme_minimal(base_size = 12))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)

# 非分组：每人一行
ungrouped <- raw %>%
  filter(PERIOD == 1, !is.na(TOTCHOL), !is.na(AGE_group)) %>%
  transmute(Y = as.integer(TOTCHOL >= 200),
            X = as.numeric(AGE_group))      # 年龄组当作分值 1/2/3
nrow(ungrouped)

# 分组：每个年龄组汇总成 成功(undesirable)/失败(desirable)
grouped <- ungrouped %>%
  group_by(X) %>%
  summarise(undesirable = sum(Y), n = n(), .groups = "drop") %>%
  mutate(desirable = n - undesirable, prop = undesirable / n)
grouped


## ----fit----------------------------------------------------------------------
# 非分组
m0_ung <- glm(Y ~ 1, data = ungrouped, family = binomial)
m1_ung <- glm(Y ~ X, data = ungrouped, family = binomial)

# 分组
m0_grp <- glm(cbind(undesirable, desirable) ~ 1, data = grouped, family = binomial)
m1_grp <- glm(cbind(undesirable, desirable) ~ X, data = grouped, family = binomial)

# M1（分组）系数：和非分组应当完全一致
summary(m1_grp)$coefficients
coef(m1_ung)


## ----dev-table----------------------------------------------------------------
dev_tab <- tibble(
  data_entry = c("Ungrouped", "Ungrouped", "Grouped", "Grouped"),
  model = c("M0", "M1", "M0", "M1"),
  residual_deviance = c(deviance(m0_ung), deviance(m1_ung),
                        deviance(m0_grp), deviance(m1_grp)),
  df_residual = c(df.residual(m0_ung), df.residual(m1_ung),
                  df.residual(m0_grp), df.residual(m1_grp))
)
dev_tab


## ----diff-table---------------------------------------------------------------
diff_tab <- tibble(
  data_entry = c("Ungrouped", "Grouped"),
  dev_M0 = c(deviance(m0_ung), deviance(m0_grp)),
  dev_M1 = c(deviance(m1_ung), deviance(m1_grp)),
  difference = dev_M0 - dev_M1,
  df = c(df.residual(m0_ung) - df.residual(m1_ung),
         df.residual(m0_grp) - df.residual(m1_grp))
)
diff_tab


## ----lrt----------------------------------------------------------------------
anova(m0_ung, m1_ung, test = "LRT")   # 非分组的似然比检验


## ----plot-fit, fig.height=4---------------------------------------------------
newd <- tibble(X = seq(1, 3, 0.05))
newd$M1 <- predict(m1_grp, newdata = newd, type = "response")

ggplot() +
  geom_point(data = grouped, aes(X, prop), size = 3, colour = "black") +
  geom_line(data = newd, aes(X, M1), colour = "firebrick", linewidth = 1) +
  geom_hline(yintercept = predict(m0_grp, type = "response")[1],
             linetype = 2, colour = "steelblue") +
  scale_x_continuous(breaks = 1:3, labels = c("≤55", "55–65", ">65")) +
  labs(title = "胆固醇不理想的概率：观测 vs Logistic 拟合",
       subtitle = "黑点=各组观测比例；红线=M1拟合；蓝虚线=M0（总平均）",
       x = "年龄组（分值 X）", y = "P(undesirable)")


## ----or-----------------------------------------------------------------------
# 斜率的优势比及 95% CI：年龄每升高一级，"不理想"的优势乘以多少
exp(cbind(OR = coef(m1_ung), confint.default(m1_ung)))


## ----linear-vs-factor---------------------------------------------------------
# 把年龄当因子（每组一个独立参数）。对分组数据，这就是"饱和模型"，偏差=0
m1_factor <- glm(cbind(undesirable, desirable) ~ factor(X), data = grouped, family = binomial)

# 线性 vs 因子：检验"线性趋势"是否有 lack-of-fit
anova(m1_grp, m1_factor, test = "LRT")


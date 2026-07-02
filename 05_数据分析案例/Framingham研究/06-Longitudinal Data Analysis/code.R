## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7.5, fig.height = 4.2,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
library(lme4)       # LMM / GLMM 的主流实现
library(geepack)    # GEE 的主流实现
theme_set(theme_minimal(base_size = 12))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)

dat <- raw %>%
  filter(!is.na(TOTCHOL), !is.na(BMI), !is.na(AGE_group), !is.na(SEX)) %>%
  mutate(
    log_chol = log(TOTCHOL),
    undesirable = as.integer(TOTCHOL >= 200),
    Age = factor(AGE_group, levels = c(1, 2, 3), labels = c("≤55", "55–65", ">65")),
    Sex = factor(SEX, levels = c(0, 1), labels = c("Men", "Women")),
    period_c = PERIOD - 2          # 时间中心化（第2期为0），减小与二次项的共线性
  ) %>%
  arrange(RANDID, PERIOD)

c(rows = nrow(dat), subjects = n_distinct(dat$RANDID))
table(dat$PERIOD)                  # 每期观测数（越往后越少：失访/死亡）


## ----traj, fig.height=4.2-----------------------------------------------------
set.seed(7)
sub <- sample(unique(dat$RANDID), 80)            # 随机抽 80 人画轨迹，避免一团乱麻

ggplot(dat %>% filter(RANDID %in% sub),
       aes(PERIOD, log_chol, group = RANDID)) +
  geom_line(alpha = 0.3, colour = "steelblue") +
  stat_summary(aes(group = 1), data = dat, fun = mean,
               geom = "line", colour = "firebrick", linewidth = 1.2) +
  stat_summary(aes(group = 1), data = dat, fun = mean,
               geom = "point", colour = "firebrick", size = 2.5) +
  scale_x_continuous(breaks = 1:3) +
  labs(title = "个体 log(胆固醇) 轨迹（蓝，随机80人）与总体均值（红）",
       x = "PERIOD（随访期）", y = "log(总胆固醇)")


## ----trend-test---------------------------------------------------------------
dat %>% group_by(PERIOD) %>%
  summarise(mean_logchol = mean(log_chol), n = n(), .groups = "drop")

# 用边际模型粗判趋势形状：线性 vs 二次
lin  <- lm(log_chol ~ period_c, data = dat)
quad <- lm(log_chol ~ period_c + I(period_c^2), data = dat)
anova(lin, quad)     # 二次项是否显著改善


## ----lmm----------------------------------------------------------------------
lmm <- lmer(log_chol ~ BMI + period_c + I(period_c^2) + Age + Sex + (1 | RANDID),
            data = dat, REML = TRUE)
round(summary(lmm)$coefficients, 5)


## ----lmm-icc------------------------------------------------------------------
# 方差成分 → 组内相关 ICC = 个体间方差 / 总方差
vc <- as.data.frame(VarCorr(lmm))
sig_b2 <- vc$vcov[vc$grp == "RANDID"]    # 个体间（随机截距）方差
sig_e2 <- vc$vcov[vc$grp == "Residual"]  # 个体内（残差）方差
c(var_between = sig_b2, var_within = sig_e2, ICC = sig_b2 / (sig_b2 + sig_e2))


## ----glmm---------------------------------------------------------------------
glmm <- glmer(undesirable ~ BMI + period_c + I(period_c^2) + Age + Sex + (1 | RANDID),
              data = dat, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
round(summary(glmm)$coefficients, 5)


## ----glmm-varcomp-------------------------------------------------------------
sig_b2_glmm <- as.data.frame(VarCorr(glmm))$vcov[1]   # 随机截距方差
sig_b2_glmm


## ----gee----------------------------------------------------------------------
gee <- geeglm(undesirable ~ BMI + period_c + I(period_c^2) + Age + Sex,
              id = RANDID, data = dat,
              family = binomial, corstr = "exchangeable")
round(summary(gee)$coefficients, 5)
summary(gee)$corr        # 估计出的工作相关系数 alpha


## ----compare------------------------------------------------------------------
# 注意：glmer 的 summary()$coefficients 是"矩阵"，geeglm 的是"数据框"，
# 列名也不同（Std. Error vs Std.err），所以分别取标量更稳妥。
est_glmm <- summary(glmm)$coefficients["BMI", "Estimate"]
se_glmm  <- summary(glmm)$coefficients["BMI", "Std. Error"]
est_gee  <- summary(gee)$coefficients["BMI", "Estimate"]
se_gee   <- summary(gee)$coefficients["BMI", "Std.err"]

cmp <- tibble(
  Model = c("GLMM（个体特异）", "GEE（总体平均）"),
  beta_BMI = c(est_glmm, est_gee),
  SE = c(se_glmm, se_gee),
  OR = exp(c(est_glmm, est_gee))
)
cmp

# 理论上的"收缩因子"：PA ≈ SS / sqrt(1 + 0.346 * sigma_b^2)
shrink <- 1 / sqrt(1 + (16 / (15 * pi))^2 * sig_b2_glmm)
c(theoretical_ratio_PA_over_SS = shrink,
  empirical_ratio = est_gee / est_glmm)


## ----gee-corr-----------------------------------------------------------------
gee_ind <- geeglm(undesirable ~ BMI + period_c + I(period_c^2) + Age + Sex,
                  id = RANDID, data = dat,
                  family = binomial, corstr = "independence")

tibble(
  corr_structure = c("independence", "exchangeable"),
  beta_BMI = c(coef(gee_ind)["BMI"], coef(gee)["BMI"]),
  SE_BMI = c(summary(gee_ind)$coefficients["BMI", "Std.err"],
             summary(gee)$coefficients["BMI", "Std.err"])
)


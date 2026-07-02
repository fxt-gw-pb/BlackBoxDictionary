## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7.5, fig.height = 4.4,
  dpi = 150, dev = "ragg_png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
library(survival)   # 生存分析的核心包：Surv() / survfit() / coxph()
library(survminer)  # ggsurvplot()：主流的 KM 曲线绘制（自动标删失点 + 风险表 + p 值）
library(broom)      # tidy() 把模型结果整理成数据框
theme_set(theme_minimal(base_size = 12, base_family = "PingFang SC"))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)
dat <- filter(raw, PERIOD == 1)          # 只取基线
c(rows = nrow(dat), deaths = sum(dat$DEATH))


## ----missing------------------------------------------------------------------
n_miss <- sum(is.na(dat$BMI))
c(n_missing_BMI = n_miss, pct = round(100 * n_miss / nrow(dat), 2))

dat$bmi_miss <- factor(ifelse(is.na(dat$BMI), "Missing", "Observed"),
                       levels = c("Observed", "Missing"))


## ----missing-km, fig.height=5.6-----------------------------------------------
fit_miss <- survfit(Surv(TIMEDTH, DEATH) ~ bmi_miss, data = dat)
survdiff(Surv(TIMEDTH, DEATH) ~ bmi_miss, data = dat)   # log-rank 检验（数值结果）

ggsurvplot(
  fit_miss, data = dat,
  censor = TRUE, censor.size = 2,          # "+" 自动标出删失点（survminer 默认开启）
  conf.int = TRUE,
  pval = TRUE, pval.method = TRUE,         # 在图上自动标注 log-rank p 值
  risk.table = TRUE, risk.table.height = 0.28,   # 底部"风险人数"表
  legend.title = "BMI", legend.labs = c("Observed", "Missing"),
  palette = c("steelblue", "firebrick"),
  xlab = "时间（天）", ylab = "生存概率",
  title = "Kaplan–Meier：BMI 缺失组 vs 观测组",
  ggtheme = theme_minimal(base_family = "PingFang SC"),
  tables.theme = theme_minimal(base_family = "PingFang SC")
)


## ----weight-------------------------------------------------------------------
dat <- dat %>%
  mutate(
    weight4 = cut(BMI, breaks = c(0, 18.5, 25, 30, Inf), right = FALSE,
                  labels = c("Underweight", "Healthy", "Overweight", "Obese")),
    weight5 = factor(ifelse(is.na(BMI), "Missing", as.character(weight4)),
                     levels = c("Underweight", "Healthy", "Overweight", "Obese", "Missing"))
  )
dat %>% group_by(weight5) %>%
  summarise(n = n(), deaths = sum(DEATH), death_rate = round(mean(DEATH), 3), .groups = "drop")


## ----weight-km, fig.height=6.2------------------------------------------------
fit_w <- survfit(Surv(TIMEDTH, DEATH) ~ weight5, data = dat)
survdiff(Surv(TIMEDTH, DEATH) ~ weight5, data = dat)

# 调色板顺序对应 weight5 的水平：Underweight/Healthy/Overweight/Obese/Missing
pal <- c("#2c7fb8", "#31a354", "#fdae6b", "#d7301f", "grey50")
ggsurvplot(
  fit_w, data = dat,
  censor = TRUE, censor.size = 2,
  pval = TRUE, pval.method = TRUE,
  risk.table = TRUE, risk.table.height = 0.32, risk.table.fontsize = 3,
  legend.title = "类别",
  legend.labs = c("Underweight", "Healthy", "Overweight", "Obese", "Missing"),
  palette = pal,
  xlab = "时间（天）", ylab = "生存概率",
  title = "Kaplan–Meier：按体重类别（4 类 + 缺失）",
  ggtheme = theme_minimal(base_family = "PingFang SC"),
  tables.theme = theme_minimal(base_family = "PingFang SC")
)


## ----median-------------------------------------------------------------------
fit_w4 <- survfit(Surv(TIMEDTH, DEATH) ~ weight4, data = dat,
                  subset = !is.na(dat$weight4))
summary(fit_w4)$table[, c("records", "events", "median")]


## ----cox----------------------------------------------------------------------
cox_dat <- dat %>%
  filter(!is.na(BMI), !is.na(GLUCOSE), !is.na(TOTCHOL), !is.na(CURSMOKE),
         !is.na(PREVHYP), !is.na(AGE_group), !is.na(SEX)) %>%
  mutate(
    bmi_c    = BMI - 25,                 # 中心化到 25
    logglu_c = log(GLUCOSE) - mean(log(GLUCOSE)),
    Sex = factor(SEX), Age = factor(AGE_group),
    Hyp = factor(PREVHYP), Smoke = factor(CURSMOKE)
  )

cox_fit <- coxph(
  Surv(TIMEDTH, DEATH) ~ bmi_c + I(bmi_c^2) + Sex +
    logglu_c + I(logglu_c^2) + Hyp * Age + Smoke + TOTCHOL,
  data = cox_dat, ties = "efron"
)
broom::tidy(cox_fit, exponentiate = TRUE, conf.int = TRUE) %>%
  select(term, HR = estimate, conf.low, conf.high, p.value) %>%
  mutate(across(where(is.numeric), ~round(., 4)))


## ----ushape-------------------------------------------------------------------
cox_no_q   <- update(cox_fit, . ~ . - I(bmi_c^2))          # 去掉二次项
anova(cox_no_q, cox_fit)                                   # U 形项是否显著

b1 <- coef(cox_fit)["bmi_c"]; b2 <- coef(cox_fit)["I(bmi_c^2)"]
bmi_vertex <- if (b2 > 0) 25 - b1 / (2 * b2) else NA       # 风险最低的 BMI
c(beta_linear = b1, beta_quad = b2, hazard_min_BMI = bmi_vertex)


## ----cox-bmi-plot, fig.height=4.2---------------------------------------------
b <- coef(cox_fit); V <- vcov(cox_fit)
grid <- tibble(BMI = seq(16, 45, 0.25), xc = BMI - 25,
               logHR = b1 * xc + b2 * xc^2)
idx <- c("bmi_c", "I(bmi_c^2)")
grid$se <- sqrt(grid$xc^2 * V[idx[1], idx[1]] +
                grid$xc^4 * V[idx[2], idx[2]] +
                2 * grid$xc^3 * V[idx[1], idx[2]])
ggplot(grid, aes(BMI, exp(logHR))) +
  geom_ribbon(aes(ymin = exp(logHR - 1.96 * se), ymax = exp(logHR + 1.96 * se)),
              fill = "steelblue", alpha = 0.15) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_hline(yintercept = 1, linetype = 2, colour = "grey40") +
  geom_vline(xintercept = 25, linetype = 3, colour = "grey60") +
  labs(title = "校正后的 BMI–死亡风险关系（参照 BMI=25）",
       x = expression(BMI~(kg/m^2)), y = "风险比 HR")


## ----interaction--------------------------------------------------------------
cox_no_int <- update(cox_fit, . ~ . - Hyp:Age)
anova(cox_no_int, cox_fit)        # 交互是否显著

# 各年龄组内"高血压 vs 无"的 HR（合并主效应 + 交互）
hr_hyp_in_age <- function(g) {
  L <- setNames(rep(0, length(b)), names(b)); L["Hyp1"] <- 1
  if (g == 2) L["Hyp1:Age2"] <- 1
  if (g == 3) L["Hyp1:Age3"] <- 1
  est <- sum(L * b); se <- sqrt(as.numeric(t(L) %*% V %*% L))
  c(HR = exp(est), low = exp(est - 1.96 * se), up = exp(est + 1.96 * se))
}
round(t(sapply(1:3, hr_hyp_in_age)), 3)


## ----zph----------------------------------------------------------------------
cox.zph(cox_fit)


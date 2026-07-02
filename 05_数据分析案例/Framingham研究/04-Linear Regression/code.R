## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7.5, fig.height = 4,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
library(broom)     # tidy() / augment()：把模型结果整理成数据框，最主流
library(car)       # leveneTest()：方差齐性检验
theme_set(theme_minimal(base_size = 12))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)

base <- raw %>%
  filter(PERIOD == 1) %>%
  distinct(RANDID, .keep_all = TRUE) %>%      # 基线每人一行
  mutate(
    log_chol = log(TOTCHOL),
    Age = factor(AGE_group, levels = c(1, 2, 3), labels = c("≤55", "55–65", ">65")),
    Sex = factor(SEX, levels = c(0, 1), labels = c("Men", "Women"))
  )


## ----lm-fit-------------------------------------------------------------------
da <- base %>% select(log_chol, BMI) %>% drop_na()
fit_bmi <- lm(log_chol ~ BMI, data = da)
summary(fit_bmi)


## ----lm-interpret-------------------------------------------------------------
b <- coef(fit_bmi)["BMI"]
c(beta = b, multiplicative = exp(b), percent_per_unit = (exp(b) - 1) * 100)


## ----lm-scatter, fig.height=4-------------------------------------------------
ggplot(da, aes(BMI, log_chol)) +
  geom_point(alpha = 0.2, colour = "#4C72B0") +
  geom_smooth(method = "lm", colour = "firebrick") +
  labs(title = "log(总胆固醇) vs BMI（含回归直线）", x = "BMI", y = "log(总胆固醇)")


## ----r2-check-----------------------------------------------------------------
r  <- cor(da$BMI, da$log_chol)        # Pearson 相关
r2 <- summary(fit_bmi)$r.squared      # 模型 R²
c(R2_model = r2, r_pearson = r, r_squared = r^2, difference = r2 - r^2)


## ----resid-diag, fig.width=8, fig.height=3.8----------------------------------
aug <- augment(fit_bmi)   # 含 .fitted（拟合值）、.resid（残差）、.std.resid（标准化残差）

p1 <- ggplot(aug, aes(.fitted, .resid)) +
  geom_point(alpha = 0.2) + geom_hline(yintercept = 0, linetype = 2) +
  geom_smooth(se = FALSE, colour = "firebrick") +
  labs(title = "残差 vs 拟合值", x = "拟合值", y = "残差")

p2 <- ggplot(aug, aes(sample = .std.resid)) +
  stat_qq(alpha = 0.2) + stat_qq_line(colour = "firebrick") +
  labs(title = "标准化残差 QQ 图", x = "理论分位数", y = "标准化残差")

patchwork::wrap_plots(p1, p2)


## ----outliers-----------------------------------------------------------------
# |标准化残差| > 3 视为可疑离群点
sum(abs(aug$.std.resid) > 3)


## ----aov-age------------------------------------------------------------------
db <- base %>% select(log_chol, Age) %>% drop_na()
fit_age <- aov(log_chol ~ Age, data = db)
summary(fit_age)


## ----aov-age-box, fig.height=4------------------------------------------------
ggplot(db, aes(Age, log_chol, fill = Age)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  labs(title = "log(总胆固醇) by 年龄组", x = "年龄组", y = "log(总胆固醇)")


## ----aov-age-diag, fig.width=8, fig.height=3.6--------------------------------
aug_age <- augment(fit_age)
p1 <- ggplot(aug_age, aes(.fitted, .resid)) +
  geom_point(alpha = 0.2, position = position_jitter(width = 0.01)) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "残差 vs 拟合值（年龄 ANOVA）", x = "拟合值（各组均值）", y = "残差")
p2 <- ggplot(aug_age, aes(sample = .std.resid)) +
  stat_qq(alpha = 0.2) + stat_qq_line(colour = "firebrick") +
  labs(title = "残差 QQ 图", x = "理论分位数", y = "标准化残差")
patchwork::wrap_plots(p1, p2)

leveneTest(log_chol ~ Age, data = db)   # H0: 各组方差相等


## ----tukey--------------------------------------------------------------------
TukeyHSD(fit_age)


## ----aov-sex------------------------------------------------------------------
dc <- base %>% select(log_chol, Sex) %>% drop_na()
fit_sex <- aov(log_chol ~ Sex, data = dc)
summary(fit_sex)

dc %>% group_by(Sex) %>%
  summarise(n = n(), mean_logchol = mean(log_chol), .groups = "drop")


## ----aov-sex-box, fig.height=3.8----------------------------------------------
ggplot(dc, aes(Sex, log_chol, fill = Sex)) +
  geom_boxplot(alpha = 0.8, show.legend = FALSE) +
  labs(title = "log(总胆固醇) by 性别", x = NULL, y = "log(总胆固醇)")


## ----aov-sex-levene-----------------------------------------------------------
leveneTest(log_chol ~ Sex, data = dc)


## ----mlr----------------------------------------------------------------------
dm <- base %>% select(log_chol, BMI, Age, Sex) %>% drop_na()
fit_main <- lm(log_chol ~ BMI + Age + Sex, data = dm)
tidy(fit_main)
glance(fit_main)$r.squared


## ----interaction--------------------------------------------------------------
fit_int <- lm(log_chol ~ BMI * Sex + BMI * Age, data = dm)
anova(fit_main, fit_int)    # 整体：加入所有交互项是否显著改善模型


## ----interaction-split--------------------------------------------------------
fit_bmiSex <- lm(log_chol ~ BMI * Sex + Age, data = dm)   # 只加 BMI×性别
fit_bmiAge <- lm(log_chol ~ BMI * Age + Sex, data = dm)   # 只加 BMI×年龄
c(BMI_x_Sex = anova(fit_main, fit_bmiSex)[["Pr(>F)"]][2],
  BMI_x_Age = anova(fit_main, fit_bmiAge)[["Pr(>F)"]][2])


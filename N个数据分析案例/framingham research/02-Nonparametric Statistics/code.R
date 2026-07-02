## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 8, fig.height = 4,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
library(boot)        # 自助法 bootstrap，最主流实现
library(nortest)     # 大样本正态性检验（Anderson–Darling / Lilliefors）
theme_set(theme_minimal(base_size = 12))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)
base <- filter(raw, PERIOD == 1)

bmi  <- base$BMI[!is.na(base$BMI)]            # 完整 BMI
chol <- base$TOTCHOL[!is.na(base$TOTCHOL)]    # 完整胆固醇
pair <- base %>% select(BMI, TOTCHOL) %>% drop_na()   # 成对完整（用于相关）

c(n_bmi = length(bmi), n_chol = length(chol), n_pair = nrow(pair))


## ----helpers------------------------------------------------------------------
# 偏度 / 峰度的小函数（峰度这里用"超额峰度"，正态=0）
skew <- function(x) mean((x - mean(x))^3) / sd(x)^3
kurt <- function(x) mean((x - mean(x))^4) / sd(x)^4 - 3


## ----bmi-test-----------------------------------------------------------------
c(mean = mean(bmi), sd = sd(bmi), skewness = skew(bmi), kurtosis = kurt(bmi))

# Shapiro–Wilk：R 中最常用的正态性检验，但样本量上限 5000
shapiro.test(bmi)


## ----bmi-raw, fig.height=3.6--------------------------------------------------
p1 <- ggplot(tibble(bmi), aes(bmi)) +
  geom_histogram(bins = 30, fill = "#4C72B0", colour = "white") +
  labs(title = "原始 BMI 直方图", x = "BMI", y = "频数")
p2 <- ggplot(tibble(bmi), aes(sample = bmi)) +
  stat_qq(alpha = 0.4) + stat_qq_line(colour = "firebrick") +
  labs(title = "原始 BMI 的 QQ 图", x = "理论分位数", y = "样本分位数")
patchwork::wrap_plots(p1, p2)


## ----logbmi-test--------------------------------------------------------------
log_bmi <- log(bmi)
c(skewness = skew(log_bmi), kurtosis = kurt(log_bmi))
shapiro.test(log_bmi)


## ----logbmi-fig, fig.height=3.6-----------------------------------------------
p1 <- ggplot(tibble(log_bmi), aes(log_bmi)) +
  geom_histogram(bins = 30, fill = "#DD8452", colour = "white") +
  labs(title = "log(BMI) 直方图", x = "log(BMI)", y = "频数")
p2 <- ggplot(tibble(log_bmi), aes(sample = log_bmi)) +
  stat_qq(alpha = 0.4) + stat_qq_line(colour = "firebrick") +
  labs(title = "log(BMI) 的 QQ 图", x = "理论分位数", y = "样本分位数")
patchwork::wrap_plots(p1, p2)


## ----chol-test----------------------------------------------------------------
c(skewness = skew(chol), kurtosis = kurt(chol))
shapiro.test(chol)

log_chol <- log(chol)
c(log_skewness = skew(log_chol), log_kurtosis = kurt(log_chol))
shapiro.test(log_chol)


## ----chol-fig, fig.height=3.6-------------------------------------------------
p1 <- ggplot(tibble(chol), aes(chol)) +
  geom_histogram(bins = 30, fill = "#55A868", colour = "white") +
  labs(title = "原始胆固醇直方图", x = "TOTCHOL", y = "频数")
p2 <- ggplot(tibble(log_chol), aes(log_chol)) +
  geom_histogram(bins = 30, fill = "#C44E52", colour = "white") +
  labs(title = "log(胆固醇) 直方图", x = "log(TOTCHOL)", y = "频数")
patchwork::wrap_plots(p1, p2)


## ----corr---------------------------------------------------------------------
ct <- cor.test(pair$BMI, pair$TOTCHOL, method = "spearman")
ct

rs <- unname(ct$estimate)        # Spearman 相关系数
n  <- nrow(pair)
se_formula <- sqrt(1 / (n - 1))  # 讲义公式：SE(r_s) ≈ sqrt(1/(n-1))
c(r_spearman = rs, n = n, SE_formula = se_formula, p_value = ct$p.value)


## ----corr-compare-------------------------------------------------------------
# 同时给出 Pearson 与 Kendall 作对比
rbind(
  Pearson  = c(est = cor(pair$BMI, pair$TOTCHOL),
               p = cor.test(pair$BMI, pair$TOTCHOL)$p.value),
  Spearman = c(est = rs, p = ct$p.value),
  Kendall  = c(est = cor(pair$BMI, pair$TOTCHOL, method = "kendall"),
               p = cor.test(pair$BMI, pair$TOTCHOL, method = "kendall")$p.value)
)


## ----corr-scatter, fig.height=4-----------------------------------------------
ggplot(pair, aes(BMI, TOTCHOL)) +
  geom_point(alpha = 0.2, colour = "#4C72B0") +
  geom_smooth(method = "lm", se = FALSE, colour = "firebrick") +
  labs(title = "BMI vs 总胆固醇（含线性趋势）", x = "BMI", y = "总胆固醇 (mg/dL)")


## ----boot---------------------------------------------------------------------
boot_rs <- function(data, idx) cor(data$BMI[idx], data$TOTCHOL[idx], method = "spearman")

set.seed(123)
bo <- boot(pair, statistic = boot_rs, R = 2000)   # 2000 次重抽样
bo
boot_se <- sd(bo$t)                                # bootstrap 标准误
c(SE_bootstrap = boot_se, SE_formula = se_formula)


## ----boot-hist, fig.height=3.8------------------------------------------------
ggplot(tibble(r = bo$t[, 1]), aes(r)) +
  geom_histogram(bins = 30, fill = "#CCB974", colour = "white") +
  geom_vline(xintercept = rs, linetype = 2, colour = "firebrick") +
  labs(title = "Spearman 相关的 Bootstrap 分布（2000 次）",
       x = "Bootstrap 相关系数", y = "频数")


## ----boot-ci------------------------------------------------------------------
boot.ci(bo, type = c("perc", "norm"))   # 百分位法 & 正态近似法的 95% CI


## ----ext1---------------------------------------------------------------------
ad.test(bmi)          # Anderson–Darling（无 n≤5000 限制）
lillie.test(bmi)      # Lilliefors（KS 的改良版）


## ----ext2---------------------------------------------------------------------
grp <- base %>%
  mutate(chol_status = if_else(TOTCHOL >= 200, "Undesirable", "Desirable")) %>%
  filter(!is.na(BMI), !is.na(TOTCHOL))

wilcox.test(BMI ~ chol_status, data = grp)   # 非参数两组比较
grp %>% group_by(chol_status) %>%
  summarise(n = n(), median_BMI = median(BMI), IQR = IQR(BMI), .groups = "drop")


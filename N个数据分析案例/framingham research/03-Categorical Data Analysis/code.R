## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.path = "figures/", fig.width = 7.5, fig.height = 4.2,
  dpi = 150, dev = "png", comment = "#>"
)


## ----prep---------------------------------------------------------------------
library(tidyverse)
theme_set(theme_minimal(base_size = 12))

raw <- read_csv("../rawdata/Framingham_data.csv", show_col_types = FALSE)

dat <- raw %>%
  filter(PERIOD == 1, !is.na(TOTCHOL), !is.na(BMI), !is.na(AGE_group), !is.na(SEX)) %>%
  mutate(
    chol = factor(if_else(TOTCHOL >= 200, "Undesirable", "Desirable"),
                  levels = c("Desirable", "Undesirable")),
    Sex  = factor(SEX, levels = c(0, 1), labels = c("Men", "Women")),
    Age  = factor(AGE_group, levels = c(1, 2, 3), labels = c("≤55", "55–65", ">65")),
    BMIcat = cut(BMI, breaks = c(-Inf, 18.5, 25, 30, Inf), right = FALSE,
                 labels = c("Underweight", "Normal", "Overweight", "Obese"))
  )
nrow(dat)   # 完整样本量


## ----sex-table----------------------------------------------------------------
tab_sex <- table(dat$Sex, dat$chol)
tab_sex
round(prop.table(tab_sex, margin = 1), 3)   # 按行（每个性别内）求比例


## ----sex-test-----------------------------------------------------------------
chisq.test(tab_sex)          # 默认带 Yates 连续性校正
chisq.test(tab_sex, correct = FALSE)$expected   # 期望频数：都远大于 5


## ----sex-bar, fig.height=3.8--------------------------------------------------
ggplot(dat, aes(Sex, fill = chol)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "胆固醇状态 by 性别", x = NULL, y = "比例", fill = "胆固醇")


## ----age-table----------------------------------------------------------------
tab_age <- table(dat$Age, dat$chol)
tab_age
round(prop.table(tab_age, 1), 3)
chisq.test(tab_age)


## ----age-trend----------------------------------------------------------------
# prop.trend.test(成功数, 总数)：检验比例随有序组的线性趋势
succ  <- tab_age[, "Undesirable"]
total <- rowSums(tab_age)
rbind(undesirable = succ, total = total, prop = round(succ / total, 3))
prop.trend.test(succ, total)


## ----age-bar, fig.height=3.8--------------------------------------------------
ggplot(dat, aes(Age, fill = chol)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "胆固醇状态 by 年龄组", x = "年龄组", y = "比例", fill = "胆固醇")


## ----bmi-table----------------------------------------------------------------
tab_bmi <- table(dat$BMIcat, dat$chol)
tab_bmi
round(prop.table(tab_bmi, 1), 3)
chisq.test(tab_bmi)


## ----bmi-expected-------------------------------------------------------------
chisq.test(tab_bmi)$expected


## ----bmi-trend----------------------------------------------------------------
succ_b  <- tab_bmi[, "Undesirable"]; total_b <- rowSums(tab_bmi)
prop.trend.test(succ_b, total_b)
round(succ_b / total_b, 3)


## ----bmi-bar, fig.height=3.8--------------------------------------------------
ggplot(dat, aes(BMIcat, fill = chol)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "胆固醇状态 by BMI 组", x = "BMI 组", y = "比例", fill = "胆固醇")


## ----summary------------------------------------------------------------------
tibble(
  Variable = c("Sex", "Age group", "BMI group"),
  Method   = c("Pearson χ²", "Pearson χ² + 趋势检验", "Pearson χ² + 趋势检验"),
  p_value  = c(chisq.test(tab_sex)$p.value,
               chisq.test(tab_age)$p.value,
               chisq.test(tab_bmi)$p.value)
)


## ----ext-or-------------------------------------------------------------------
# 性别的 OR：以"undesirable"为事件，比较 Women vs Men
fisher.test(tab_sex)$estimate          # 条件极大似然 OR
fisher.test(tab_sex)$conf.int          # 95% 置信区间


## ----ext-cmh------------------------------------------------------------------
# 把 BMI 二分（Overweight/Obese vs 其余）便于构造 2×2×2 分层表
dat2 <- dat %>% mutate(highBMI = factor(BMI >= 25, labels = c("BMI<25", "BMI≥25")))
strat <- table(dat2$highBMI, dat2$chol, dat2$Sex)   # 2×2×2：按性别分层
strat
mantelhaen.test(strat)                  # 控制性别后，高 BMI 与胆固醇是否仍相关


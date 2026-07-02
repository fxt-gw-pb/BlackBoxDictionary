---
title: XGBoost时间序列预测
english_name: XGBoost for Time Series Forecasting
slug: xgboost-for-time-series-forecasting
aliases: [XGBoost time series, xgboost forecasting, "XGBoost时间序列预测（XGBoost for Time Series Forecasting）"]
category: 时间序列与时序建模
subcategory: 机器学习时序预测
tags: [医学统计, 数据科学, 时间序列, 预测, XGBoost, 特征工程]
status: 已建
difficulty: intermediate
question_type: 基于滞后特征的非线性时间序列预测
data_type: [时间序列数据, 表格数据, 多变量时间序列数据]
outcome_type: [连续型, 时间序列, 二分类]
python_packages: [xgboost, scikit-learn]
r_packages: [xgboost]
---

# XGBoost时间序列预测（XGBoost for Time Series Forecasting）

## 1. 方法概览

### 1.1 定义

XGBoost 时间序列预测是把时间序列问题转换为监督学习表格问题，再用 XGBoost 建模。核心做法是构造滞后特征、滚动统计量、日历特征和外部变量，用它们预测未来目标值。

### 1.2 它主要解决什么问题

- 研究问题：时间序列中存在非线性、交互和外部变量影响时，如何做高性能预测。
- 适用任务：单步或多步预测、带外部变量的运营预测、非线性时序回归或分类。
- 常见医学场景：每日就诊量预测、药品需求预测、患者风险随时间更新、可穿戴设备特征预测。

### 1.3 直觉理解

XGBoost 本身不懂时间顺序，所以我们要把“过去”显式做成特征：昨天的值、过去 7 天均值、星期几、节假日、天气等。模型再从这些特征里学习非线性预测规则。

## 2. 数学形式

### 2.1 核心公式

先把序列转换为监督学习样本：

$$
X_t=[y_{t-1},y_{t-2},\dots,y_{t-p}, r_t, z_t]
$$

其中 $r_t$ 是滚动统计量，$z_t$ 是日历或外部变量。预测目标为：

$$
\hat y_t=F(X_t)=\sum_{k=1}^{K}f_k(X_t)
$$

XGBoost 的正则化目标为：

$$
\mathcal{L}=\sum_{i}l(y_i,\hat y_i)+\sum_{k}\Omega(f_k)
$$

其中：

$$
\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2
$$

### 2.2 参数或统计量含义

- 滞后特征：过去若干期的目标值。
- 滚动统计量：过去窗口内均值、最大值、标准差等。
- `n_estimators`：树的数量。
- `max_depth`：单棵树最大深度。
- `learning_rate`：每棵树的贡献缩放。
- `subsample` 和 `colsample_bytree`：样本和特征采样比例。

### 2.3 关键假设

- 未来可由历史滞后、日历和外部变量特征预测。
- 训练和预测期间的特征生成逻辑一致。
- 时间切分严格避免未来信息泄露。
- 若做多步预测，需要明确递归预测或直接多步建模策略。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：滞后特征、滚动特征、日历特征、外部变量。
- 因变量类型：连续型未来值，也可为未来事件分类标签。
- 数据结构：由时间序列展开成监督学习宽表。
- 是否适合高维数据：适合中高维表格特征，但需防止过拟合。
- 是否适合缺失较多数据：XGBoost 可处理部分缺失，但时间特征缺失仍需谨慎。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：可用于多实体面板序列，但需按实体构造滞后并设计分组验证。

### 3.2 示例表格

以每日急诊量预测为例：

| Date | y | lag_1 | lag_7 | roll_mean_7 | weekday | temperature |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01-08 | 230 | 225 | 212 | 221.4 | 3 | 4.2 |
| 2026-01-09 | 241 | 230 | 225 | 224.0 | 4 | 3.8 |
| 2026-01-10 | 220 | 241 | 231 | 226.9 | 5 | 5.1 |
| 2026-01-11 | 208 | 220 | 220 | 226.0 | 6 | 6.3 |

### 3.3 输入与产出

#### 输入

- 输入数据：带时间索引的序列，以及构造好的监督学习特征表。
- 关键变量：滞后窗口、滚动窗口、预测步长、树模型超参数。
- 需要预处理的内容：按时间切分、滞后特征生成、外部变量对齐、缺失处理、时间序列交叉验证。

#### 产出

- 模型对象/统计结果：训练好的 XGBoost 模型、特征重要性、预测值。
- 参数估计：树结构和叶节点权重。
- 预测结果：未来值点预测；可用分位数模型或 bootstrap 近似区间。
- 不确定性指标：时间序列交叉验证误差、残差分布、分位数预测区间。

## 4. 适用场景

- 适合：非线性强、有多种外部变量、特征工程充分、预测表现优先的场景。
- 不适合：样本很少、需要参数解释、长期外推趋势、未来外部变量无法获得的场景。
- 使用前需要特别检查的点：滞后特征是否泄露未来、验证切分是否按时间、未来外部变量是否可用。

## 5. 实现

### 5.1 Python

常用包：

- `xgboost`
- `scikit-learn`

```python
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("daily_ed_visits_with_weather.csv", parse_dates=["Date"])
df = df.sort_values("Date")
df["lag_1"] = df["EDVisits"].shift(1)
df["lag_7"] = df["EDVisits"].shift(7)
df["roll_mean_7"] = df["EDVisits"].shift(1).rolling(7).mean()
df["weekday"] = df["Date"].dt.weekday
df = df.dropna()

features = ["lag_1", "lag_7", "roll_mean_7", "weekday", "Temperature"]
train = df[df["Date"] < "2026-06-01"]
test = df[df["Date"] >= "2026-06-01"]

model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(train[features], train["EDVisits"])
pred = model.predict(test[features])
print("MAE:", mean_absolute_error(test["EDVisits"], pred))
```

### 5.2 R

常用包：

- `xgboost`

```r
library(xgboost)

df <- df[order(df$Date), ]
df$lag_1 <- dplyr::lag(df$EDVisits, 1)
df$lag_7 <- dplyr::lag(df$EDVisits, 7)
df$roll_mean_7 <- zoo::rollmean(dplyr::lag(df$EDVisits, 1), 7, fill = NA, align = "right")
df$weekday <- as.POSIXlt(df$Date)$wday
df <- na.omit(df)

features <- c("lag_1", "lag_7", "roll_mean_7", "weekday", "Temperature")
train <- df[df$Date < as.Date("2026-06-01"), ]
test <- df[df$Date >= as.Date("2026-06-01"), ]

dtrain <- xgb.DMatrix(as.matrix(train[, features]), label = train$EDVisits)
fit <- xgboost(data = dtrain, nrounds = 300, max_depth = 3, eta = 0.05, objective = "reg:squarederror")
pred <- predict(fit, as.matrix(test[, features]))
mean(abs(test$EDVisits - pred))
```

## 6. 结果如何解释

- 核心结果看什么：时间外测试误差、特征重要性、残差随时间变化、不同预测步长表现。
- 每个主要参数如何解释：`max_depth` 控制交互复杂度，过深更容易过拟合时间噪声。
- 临床或医学意义如何表达：可用于高性能运营预测，但解释应以特征贡献和误差分析为主。
- 常见误读：特征重要性高不代表因果影响，尤其滞后值和外部变量可能共同反映趋势。

## 7. 推荐可视化

- 实际值与预测值时间曲线。
- 残差随时间变化图。
- 特征重要性或 SHAP 图。
- 不同预测 horizon 的误差曲线。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉非线性和特征交互。
- 可融合日历、天气、节假日和运营变量。
- 对表格型时间序列预测常有强表现。

### 局限

- 不内置时间结构，依赖特征工程。
- 长期外推能力弱。
- 预测区间和不确定性需要额外方法。

### 常见坑

- 滚动均值没有先 shift，导致使用当天目标值预测当天。
- 随机划分训练测试集。
- 未来不可获得的外部变量被用于预测。

## 9. 与相近方法的区别

- 和 [[XGBoost（Extreme Gradient Boosting, XGBoost）]] 的区别：通用 XGBoost 条目讲梯度提升树本身，本条目关注如何把时间序列构造成监督学习问题。
- 和 [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]] 的区别：ARIMA 是参数化时间序列模型，XGBoost 通过特征工程学习非线性映射。
- 和 [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]] 的区别：LSTM 直接读取序列窗口，XGBoost 读取手工构造的表格特征。

## 10. 医学研究中的典型应用

- 门急诊量、床位需求和检查量预测。
- 药品、耗材和血液制品需求预测。
- 多源特征驱动的患者短期风险动态预测。

## 11. 相关方法

- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[自回归积分滑动平均模型（Autoregressive Integrated Moving Average Model, ARIMA）]]
- [[LSTM时间序列预测（Long Short-Term Memory for Time Series Forecasting）]]
- [[Prophet时间序列模型（Prophet Forecasting Model）]]

## 12. 参考资料

- Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *KDD*. 2016:785-794.
- Hyndman RJ, Athanasopoulos G. *Forecasting: Principles and Practice*. 3rd ed. OTexts; 2021. [https://otexts.com/fpp3/](https://otexts.com/fpp3/) （访问日期：2026-07-02）
- XGBoost Developers. XGBoost Documentation. [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/) （访问日期：2026-07-02）

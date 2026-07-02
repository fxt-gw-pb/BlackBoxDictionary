import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# 设置 Seaborn 风格，使图表更美观
sns.set(style="whitegrid", palette="pastel")

# 读取数据集
file_path = "./dataset/005/all_stocks_5yr.csv"
df = pd.read_csv(file_path)

# 显示前几行数据
df.head()

# 查看数据集的基本信息
df.info()

# 查看数据的统计摘要
df.describe()

# 将 date 列转换为 datetime 格式
df["date"] = pd.to_datetime(df["date"])

# 检查转换后的数据类型
df.dtypes

# 检查缺失值
missing_values = df.isnull().sum()
print("缺失值情况：\n", missing_values)

# 由于股票数据缺失值可能影响分析，采用前向填充填补缺失值
df.fillna(method="ffill", inplace=True)

# 计算价格的 z-score 来检测异常值
price_columns = ["open", "high", "low", "close"]

# 计算每列的 z-score
df_zscore = df[price_columns].apply(zscore)

# 计算每行的最大 z-score 作为异常值指标
df["z_score"] = df_zscore.abs().max(axis=1)

# 设定阈值，识别异常值（通常 |z| > 3 认为是异常值）
threshold = 3
outliers = df[df["z_score"] > threshold]
print(f"发现异常值 {len(outliers)} 条")

# 处理异常值：这里我们选择用中位数替换
for col in price_columns:
    median_value = df[col].median()
    df.loc[df["z_score"] > threshold, col] = median_value

# 删除辅助列
df.drop(columns=["z_score"], inplace=True)

# 检查重复值
duplicates = df.duplicated().sum()
print(f"重复数据行数：{duplicates}")

# 删除重复值
df.drop_duplicates(inplace=True)

# 选择 AAPL 股票数据
df_aapl = df[df["Name"] == "AAPL"]

# 绘制收盘价趋势
plt.figure(figsize=(12, 6))
plt.plot(df_aapl["date"], df_aapl["close"], label="Close Price", color="dodgerblue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("AAPL Stock Closing Price Trend")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["close"], bins=50, kde=True, color="royalblue")
plt.xlabel("Stock Closing Price")
plt.ylabel("Frequency")
plt.title("Distribution of Closing Prices")
plt.show()

plt.figure(figsize=(12, 6))

# 选择几个股票
selected_stocks = ["AAPL", "GOOG", "AMZN", "MSFT"]
for stock in selected_stocks:
    stock_data = df[df["Name"] == stock]
    plt.plot(stock_data["date"], stock_data["close"], label=stock)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Trends of Selected Companies")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="Name", y="volume", data=df[df["Name"].isin(selected_stocks)], palette="Set2")
plt.xticks(rotation=90)
plt.xlabel("Stock")
plt.ylabel("Trading Volume")
plt.title("Trading Volume Distribution of Selected Stocks")
plt.show()

# 计算价格与交易量之间的相关性
corr_matrix = df[["open", "high", "low", "close", "volume"]].corr()
print("相关性矩阵：\n", corr_matrix)

# 可视化相关性矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Stock Prices and Volume")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["volume"], y=df["close"], alpha=0.5, color="blue")
plt.xlabel("Trading Volume")
plt.ylabel("Closing Price")
plt.title("Trading Volume vs. Stock Closing Price")
plt.show()

# 计算 5 日和 20 日移动均线
df["MA5"] = df.groupby("Name")["close"].transform(lambda x: x.rolling(window=5).mean())
df["MA20"] = df.groupby("Name")["close"].transform(lambda x: x.rolling(window=20).mean())


def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


df["RSI14"] = df.groupby("Name")["close"].transform(lambda x: compute_rsi(x, window=14))

df["volatility"] = df.groupby("Name")["close"].transform(lambda x: x.rolling(window=5).std())

# 提取年份、月份、星期几
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek  # 0: Monday, 6: Sunday

# 选择 AAPL 股票数据
df_aapl = df[df["Name"] == "AAPL"].dropna()  # 移除 NaN 值

# 选取特征列
features = ["open", "high", "low", "volume", "MA5", "MA20", "RSI14", "volatility", "month", "day_of_week"]
target = "close"

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_aapl[features], df_aapl[target], test_size=0.2, random_state=42)

print(f"训练集样本数：{X_train.shape[0]}")
print(f"测试集样本数：{X_test.shape[0]}")

plt.figure(figsize=(12, 6))
plt.plot(df_aapl["date"], df_aapl["close"], label="Close Price", color="dodgerblue")
plt.plot(df_aapl["date"], df_aapl["MA5"], label="MA5", color="orange")
plt.plot(df_aapl["date"], df_aapl["MA20"], label="MA20", color="red")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("AAPL Stock Closing Price with Moving Averages")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df_aapl["date"], df_aapl["RSI14"], label="RSI14", color="purple")
plt.axhline(70, linestyle="--", color="red")  # 超买线
plt.axhline(30, linestyle="--", color="green")  # 超卖线
plt.xlabel("Date")
plt.ylabel("RSI Value")
plt.title("AAPL RSI Trend")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_aapl["volume"], y=df_aapl["volatility"], alpha=0.5, color="blue")
plt.xlabel("Trading Volume")
plt.ylabel("Volatility")
plt.title("Trading Volume vs. Stock Volatility")
plt.show()

# 假设 df 是处理过的股票数据，X 是特征，y 是目标（股价）
X = df.drop(columns=["close", "date", "Name"])  # 特征
y = df["close"]  # 目标变量

# 分割数据集，80%用于训练，20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost回归模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                             colsample_bytree=0.3,
                             learning_rate=0.1,
                             max_depth=5,
                             alpha=10,
                             n_estimators=1000)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测测试集
y_pred = xgb_model.predict(X_test)

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# 计算 R²
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 设置参数范围
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 500, 1000],
    "subsample": [0.7, 0.8, 1.0]
}

# 创建XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best Parameters:", grid_search.best_params_)

# 设置参数范围
param_dist = {
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# 创建XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# 使用随机搜索进行超参数调优
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=10, cv=3,
                                   scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

# 输出最优参数
print("Best Parameters:", random_search.best_params_)

# 可视化实际值与预测值的对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # 绘制y=x线作为参考
plt.title("Actual vs Predicted Stock Prices", fontsize=16)
plt.xlabel("Actual Prices", fontsize=12)
plt.ylabel("Predicted Prices", fontsize=12)
plt.show()
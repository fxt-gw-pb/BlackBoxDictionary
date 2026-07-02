import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# 加载数据集
file_path = './dataset/001/GlobalTemperatures.csv'
data = pd.read_csv(file_path)

# 查看数据基本信息
print("数据集基本信息：")
print(data.info())
print("\n数据集前5行：")
print(data.head())

# 检查数据的描述性统计
print("\n数据集的描述性统计：")
print(data.describe())

# 将日期列转换为datetime格式
data['dt'] = pd.to_datetime(data['dt'])

# 设置日期为索引
data.set_index('dt', inplace=True)

# 绘制全球地表平均气温的时间趋势
plt.figure(figsize=(16, 8))
plt.plot(data.index, data['LandAverageTemperature'], color='dodgerblue', label='Land Average Temperature')
plt.title('Land Average Temperature Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 绘制直方图和核密度估计图
plt.figure(figsize=(10, 6))
sns.histplot(data['LandAverageTemperature'].dropna(), kde=True, color='skyblue', bins=30)
plt.title('Distribution of Land Average Temperature', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.show()

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# 检查缺失值情况
print("每列缺失值数量：")
print(data.isnull().sum())

# 可视化缺失值分布
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap', fontsize=16)
plt.show()

# 填充缺失值（以插值为例）
data_cleaned = data.copy()
data_cleaned.interpolate(method='linear', inplace=True)  # 线性插值填补缺失值

# 对 LandAndOceanAverageTemperature 进行前向填充和后向填充
data_cleaned['LandAndOceanAverageTemperature'] = data_cleaned['LandAndOceanAverageTemperature'].fillna(method='ffill')
data_cleaned['LandAndOceanAverageTemperature'] = data_cleaned['LandAndOceanAverageTemperature'].fillna(method='bfill')

print("\n缺失值处理后，仍有缺失值的列：")
print(data_cleaned.isnull().sum())

# 使用箱线图检测异常值
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_cleaned[['LandAverageTemperature', 'LandMaxTemperature', 'LandMinTemperature']], palette='Set2')
plt.title('Boxplot for Temperature Variables', fontsize=16)
plt.show()

# 计算上下四分位范围（IQR）来标记异常值
Q1 = data_cleaned['LandAverageTemperature'].quantile(0.25)
Q3 = data_cleaned['LandAverageTemperature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 标记异常值
outliers = data_cleaned[(data_cleaned['LandAverageTemperature'] < lower_bound) |
                        (data_cleaned['LandAverageTemperature'] > upper_bound)]
print(f"检测到 {len(outliers)} 个异常值。")

# 处理异常值：使用上下限值进行截断（Winsorization）
data_cleaned['LandAverageTemperature'] = np.clip(data_cleaned['LandAverageTemperature'], lower_bound, upper_bound)

# 检查重复数据
duplicates = data_cleaned.duplicated().sum()
print(f"检测到 {duplicates} 条重复记录。")

# 删除重复数据
data_cleaned.drop_duplicates(inplace=True)
print(f"清理后数据集大小：{data_cleaned.shape}")

# 检查清洗后数据的描述性统计
print("\n清洗后数据的描述性统计：")
print(data_cleaned.describe())

# 绘制清洗后的时间趋势图，确认数据合理性
plt.figure(figsize=(16, 8))
plt.plot(data_cleaned.index, data_cleaned['LandAverageTemperature'], color='dodgerblue',
         label='Cleaned Land Average Temperature')
plt.title('Cleaned Land Average Temperature Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 选择相关特征
selected_features = ['LandAverageTemperature', 'LandAndOceanAverageTemperature']
data_features = data_cleaned[selected_features].copy()
print("选定的特征：", selected_features)

# 从索引中提取时间特征
data_features['Year'] = data_cleaned.index.year
data_features['Month'] = data_cleaned.index.month
data_features['Quarter'] = data_cleaned.index.quarter

# 确认提取的时间特征
print(data_features.head())

# 构造年度平均温度变化率
data_features['TemperatureChangeRate'] = data_features['LandAverageTemperature'].diff()

# 构造温度波动范围
data_features['TemperatureUncertaintyRange'] = (
    data_cleaned['LandAverageTemperatureUncertainty'] +
    data_cleaned['LandAndOceanAverageTemperatureUncertainty']
)

# 确认新特征
print("\n构造的新特征：")
print(data_features[['TemperatureChangeRate', 'TemperatureUncertaintyRange']].head())

# 可视化年度平均温度变化率
plt.figure(figsize=(16, 8))
plt.plot(data_features.index, data_features['TemperatureChangeRate'], color='purple', label='Temperature Change Rate')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Annual Temperature Change Rate', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Change Rate (°C)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 可视化温度波动范围
plt.figure(figsize=(16, 8))
plt.plot(data_features.index, data_features['TemperatureUncertaintyRange'], color='orange',
         label='Temperature Uncertainty Range')
plt.title('Temperature Uncertainty Range Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Uncertainty Range (°C)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 确认数据按时间排序
data_features.sort_index(inplace=True)

# 确定分割比例（80% 训练，20% 测试）
split_ratio = 0.8
split_index = int(len(data_features) * split_ratio)

# 划分训练集和测试集
train_data = data_features.iloc[:split_index]
test_data = data_features.iloc[split_index:]

print(f"训练集大小: {train_data.shape}")
print(f"测试集大小: {test_data.shape}")

# 可视化训练集和测试集的划分
plt.figure(figsize=(16, 8))
plt.plot(train_data.index, train_data['LandAverageTemperature'], color='blue', label='Training Data')
plt.plot(test_data.index, test_data['LandAverageTemperature'], color='red', label='Testing Data')
plt.axvline(train_data.index[-1], color='black', linestyle='--', label='Split Point')
plt.title('Training and Testing Data Split', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 确认训练集和测试集的时间范围
print(f"训练集时间范围: {train_data.index.min()} - {train_data.index.max()}")
print(f"测试集时间范围: {test_data.index.min()} - {test_data.index.max()}")

# 对 LandAverageTemperature 进行趋势和季节性分解
result_land = seasonal_decompose(data_features['LandAverageTemperature'], model='additive', period=12)

# 对 LandAndOceanAverageTemperature 进行趋势和季节性分解
result_ocean = seasonal_decompose(data_features['LandAndOceanAverageTemperature'], model='additive', period=12)

# 可视化分解结果
result_land.plot()
plt.suptitle('Decomposition of Land Average Temperature', fontsize=16)
plt.show()

result_ocean.plot()
plt.suptitle('Decomposition of Land and Ocean Average Temperature', fontsize=16)
plt.show()

# ADF 检验
adf_result = adfuller(data_features['LandAverageTemperature'].dropna())
print("ADF 检验结果：", adf_result)

# 绘制 ACF 和 PACF 图
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(data_features['LandAverageTemperature'].dropna(), lags=40, ax=plt.gca())
plt.title('ACF - LandAverageTemperature')

plt.subplot(122)
plot_pacf(data_features['LandAverageTemperature'].dropna(), lags=40, ax=plt.gca())
plt.title('PACF - LandAverageTemperature')
plt.show()

# SARIMA 模型训练
model = SARIMAX(data_features['LandAverageTemperature'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
sarima_result = model.fit()

# 模型结果总结
print(sarima_result.summary())

# 选择训练数据（以 'LandAverageTemperature' 为例）
train_series = data_features['LandAverageTemperature']

# 创建并训练 SARIMA 模型
model = SARIMAX(train_series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
sarima_result = model.fit()

# 输出模型结果
print(sarima_result.summary())

# 使用训练数据进行预测（训练集内预测）
pred_train = sarima_result.predict(start=0, end=len(train_series) - 1)

# 可视化训练数据与预测值
plt.figure(figsize=(10, 6))
plt.plot(train_series.index, train_series, label='Actual Land Average Temperature', color='blue')
plt.plot(train_series.index, pred_train, label='Predicted Land Average Temperature', color='red', linestyle='--')
plt.title('Land Average Temperature: Actual vs Predicted (Training Data)', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend()
plt.show()

# 计算 MAE 和 RMSE
mae = mean_absolute_error(train_series, pred_train)
rmse = np.sqrt(mean_squared_error(train_series, pred_train))
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 预测未来 10 年的数据（120个月）
forecast_steps = 120
forecast = sarima_result.get_forecast(steps=forecast_steps)

# 获取预测的均值和置信区间
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# 为预测结果创建连续的日期索引
forecast_index = pd.date_range(start=train_series.index[-1] + pd.DateOffset(months=1),
                               periods=forecast_steps, freq='MS')

# 可视化未来预测
plt.figure(figsize=(10, 6))
plt.plot(train_series.index, train_series, label='Actual Land Average Temperature', color='blue')
plt.plot(forecast_index, forecast_mean, label='Forecasted Land Average Temperature', color='green', linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1],
                 color='green', alpha=0.2)
plt.title('Land Average Temperature: Forecasted vs Actual', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend()
plt.show()

# 使用 auto_arima 自动选择最佳模型
auto_model = auto_arima(train_series, seasonal=True, m=12, trace=True, suppress_warnings=True)
print(auto_model.summary())

# 使用最佳模型进行预测
forecast_auto = auto_model.predict(n_periods=forecast_steps)

# 可视化优化后的预测结果
plt.figure(figsize=(10, 6))
plt.plot(train_series.index, train_series, label='Actual Land Average Temperature', color='blue')
plt.plot(forecast_index, forecast_auto, label='Optimized Forecasted Land Average Temperature', color='orange', linestyle='--')
plt.title('Optimized Forecasted Land Average Temperature', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend()
plt.show()

# 可视化对比优化前后预测结果
plt.figure(figsize=(10, 6))
plt.plot(train_series.index, train_series, label='Actual Land Average Temperature', color='blue')
plt.plot(forecast_index, forecast_mean, label='Forecasted (SARIMA)', color='green', linestyle='--')
plt.plot(forecast_index, forecast_auto, label='Optimized (auto_arima)', color='orange', linestyle='--')
plt.title('Comparison of Forecasted Land Average Temperature', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend()
plt.show()
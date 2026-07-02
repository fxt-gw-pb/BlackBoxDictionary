import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from scipy.stats import uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = "./dataset/003/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(file_path)

# 显示数据基本信息
print("数据集基本信息：")
print(df.info())

# 显示数据的前5行
print("数据样例：")
print(df.head())

# 统计描述性信息
print("数据描述性统计信息：")
print(df.describe())

# 计算缺失值数量和占比
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

# 只显示有缺失值的列
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)

# 绘制缺失值可视化
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_data.index, y=missing_data['Percentage'], palette="coolwarm")
plt.xticks(rotation=90)
plt.ylabel("Percentage of Missing Values")
plt.title("Missing Values Percentage per Column")
plt.show()

# 数值型特征：用中位数填充
num_features = df.select_dtypes(include=['number']).columns
for col in num_features:
    df[col].fillna(df[col].median(), inplace=True)

# 类别型特征：用 "None" 填充
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col].fillna("None", inplace=True)

# 确保无缺失值
print("缺失值处理后：")
print(df.isnull().sum().sum())  # 应该输出 0

# 选取数值型变量
numeric_columns = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'LotArea']

# 绘制箱线图
plt.figure(figsize=(15, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col], palette="coolwarm")
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 删除异常值
df = df[df['GrLivArea'] < 4000]
df = df[df['LotArea'] < 100000]

# 重新绘制箱线图，检查异常值处理效果
plt.figure(figsize=(15, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(y=df[col], palette="coolwarm")
    plt.title(f"Boxplot of {col} after Cleaning")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['SalePrice'], bins=50, kde=True, color='royalblue')
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.title("Distribution of Sale Prices")
plt.show()

# 只选择数值型特征进行相关性计算
numeric_df = df.select_dtypes(include=['number'])

# 计算相关性矩阵
correlation_matrix = numeric_df.corr()

# 选择与 SalePrice 相关性最高的前10个特征
top_corr_features = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)[1:11]

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df[top_corr_features.index].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Top 10 Feature Correlations with Sale Price")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], color='dodgerblue', alpha=0.6)
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.title("Living Area vs. Sale Price")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['OverallQual'], y=df['SalePrice'], palette="coolwarm")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.title("Overall Quality vs. Sale Price")
plt.show()

plt.figure(figsize=(12, 5))
sns.scatterplot(x=df['YearBuilt'], y=df['SalePrice'], color='seagreen', alpha=0.6)
plt.xlabel("Year Built")
plt.ylabel("Sale Price")
plt.title("Year Built vs. Sale Price")
plt.show()

# 统计重复行数
duplicate_count = df.duplicated().sum()
print(f"重复行数: {duplicate_count}")

# 删除重复行
df = df.drop_duplicates()

# 再次检查重复数据
print(f"去重后重复行数: {df.duplicated().sum()}")

# 替换 inf 和 NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 处理数值型列：用中位数填充
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 处理类别型列：用众数填充
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 确保所有缺失值已处理
print("缺失值总数:", df.isnull().sum().sum())  # 应该输出 0

# 选取类别型变量
categorical_cols = df.select_dtypes(include=['object']).columns

# 对于有序类别变量，使用 Label Encoding
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC']

for col in ordinal_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# 其余类别变量使用 One-Hot Encoding
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col not in ordinal_features], drop_first=True)

# 显示转换后的数据
print("类别变量转换后数据形状:", df.shape)

# 计算数值型变量的偏度
numeric_cols = df.select_dtypes(include=['number']).columns
skewed_features = df[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

# 选出偏态特征（偏度大于 0.75）
high_skew = skewed_features[skewed_features > 0.75].index
print("偏态特征:", list(high_skew))

# 对高度偏态的变量进行 Log(1+x) 变换
for col in high_skew:
    df[col] = np.log1p(df[col])

# 目标变量 SalePrice 也进行对数变换
df['SalePrice'] = np.log1p(df['SalePrice'])

# 计算相关性
correlation_matrix = df.corr()
top_features = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)[1:21]  # 选择前 20 个相关性特征

# 保留这些特征
selected_features = list(top_features.index)
df_selected = df[selected_features + ['SalePrice']]  # 加入目标变量

print("选择的特征:", selected_features)
print("特征工程后数据形状:", df_selected.shape)

# 标准化数值特征（不包括目标变量）
scaler = StandardScaler()
df_selected[selected_features] = scaler.fit_transform(df_selected[selected_features])

# 显示标准化后的数据
print(df_selected.head())
#
# 设定特征变量和目标变量
X = df_selected.drop(columns=['SalePrice'])
y = df_selected['SalePrice']

# 划分数据集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 初始化梯度提升回归模型
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
gbr_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_gbr = gbr_model.predict(X_test)

# 计算均方误差（MSE）
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)

print(f"Gradient Boosting Regression RMSE: {rmse_gbr}")

# 计算 MSE 和 RMSE
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)

# 计算 R²（决定系数）
r2_gbr = r2_score(y_test, y_pred_gbr)

print(f"Gradient Boosting Regression MSE: {mse_gbr}")
print(f"Gradient Boosting Regression RMSE: {rmse_gbr}")
print(f"Gradient Boosting Regression R²: {r2_gbr}")

# 设置参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# 初始化梯度提升回归模型
gbr_model = GradientBoostingRegressor(random_state=42)

# 使用网格搜索调参
grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                           verbose=1, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数和最优评分
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# 设置参数分布
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': [3, 4, 5]
}

# 初始化梯度提升回归模型
gbr_model = GradientBoostingRegressor(random_state=42)

# 使用随机搜索调参
random_search = RandomizedSearchCV(estimator=gbr_model, param_distributions=param_dist, n_iter=10, cv=5,
                                   scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)

# 训练模型
random_search.fit(X_train, y_train)

# 输出最优参数和最优评分
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_}")

import matplotlib.pyplot as plt

# 绘制真实值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_gbr, color='b', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted House Prices')
plt.show()

# 计算残差
residuals = y_test - y_pred_gbr

# 绘制残差图
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_gbr, residuals, color='b', alpha=0.6)
plt.hlines(y=0, xmin=y_pred_gbr.min(), xmax=y_pred_gbr.max(), color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted House Prices')
plt.show()

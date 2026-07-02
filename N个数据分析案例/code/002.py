import ast
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

# 读取数据集
movies_metadata = pd.read_csv('dataset/002/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('dataset/002/ratings_small.csv')
credits = pd.read_csv('dataset/002/credits.csv')
keywords = pd.read_csv('dataset/002/keywords.csv')

# 查看movies_metadata数据集的基本信息
print(movies_metadata.info())
print(movies_metadata.head())

# 检查movies_metadata数据集中缺失值的情况
print(movies_metadata.isnull().sum())

# 删除不必要的列，如 'homepage'，'tagline'，这些列信息不影响票房分析
movies_metadata.drop(columns=['homepage', 'tagline'], inplace=True)

# 对关键数值型列进行缺失值填充
movies_metadata['budget'].fillna(0, inplace=True)
movies_metadata['revenue'].fillna(0, inplace=True)
movies_metadata['runtime'].fillna(movies_metadata['runtime'].median(), inplace=True)

# 对分类变量进行缺失值填充
movies_metadata['status'].fillna('Unknown', inplace=True)

# 转换 id 为字符串类型
movies_metadata['id'] = movies_metadata['id'].astype(str)

# 处理日期格式
movies_metadata['release_date'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce')

# 处理 budget 和 revenue 的数据类型为整数
movies_metadata['budget'] = pd.to_numeric(movies_metadata['budget'], errors='coerce').fillna(0).astype(int)
movies_metadata['revenue'] = pd.to_numeric(movies_metadata['revenue'], errors='coerce').fillna(0).astype(int)

print(movies_metadata[['budget', 'revenue', 'runtime', 'vote_average', 'vote_count']].describe())

# 设置图形风格
sns.set_style("darkgrid")

# 绘制预算和票房收入的分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 预算分布
sns.histplot(movies_metadata[movies_metadata['budget'] > 0]['budget'], bins=50, kde=True, ax=axes[0], color="blue")
axes[0].set_title("Distribution of Movie Budgets")
axes[0].set_xlabel("Budget")
axes[0].set_ylabel("Count")

# 票房收入分布
sns.histplot(movies_metadata[movies_metadata['revenue'] > 0]['revenue'], bins=50, kde=True, ax=axes[1], color="red")
axes[1].set_title("Distribution of Movie Revenues")
axes[1].set_xlabel("Revenue")
axes[1].set_ylabel("Count")

plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=movies_metadata['vote_average'], y=movies_metadata['revenue'], alpha=0.5)
plt.title("Vote Average vs Revenue")
plt.xlabel("Vote Average")
plt.ylabel("Revenue")
plt.show()


# 解析电影类型


def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres]
    except:
        return []


movies_metadata['genres'] = movies_metadata['genres'].apply(extract_genres)

# 计算不同电影类型的平均票房
genre_revenue = {}
for genres, revenue in zip(movies_metadata['genres'], movies_metadata['revenue']):
    for genre in genres:
        if genre in genre_revenue:
            genre_revenue[genre].append(revenue)
        else:
            genre_revenue[genre] = [revenue]

# 计算平均票房
genre_avg_revenue = {genre: np.mean(revenues) for genre, revenues in genre_revenue.items()}

# 绘制柱状图
plt.figure(figsize=(12, 6))
sns.barplot(x=list(genre_avg_revenue.keys()), y=list(genre_avg_revenue.values()), palette="viridis")
plt.xticks(rotation=45)
plt.title("Average Revenue by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Revenue")
plt.show()

# 选择与票房预测相关的字段
selected_columns = ['budget', 'runtime', 'vote_average', 'vote_count', 'release_date', 'genres', 'production_companies',
                    'revenue']
movies_data = movies_metadata[selected_columns].copy()


# 解析JSON格式的电影类型
def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres]
    except:
        return []


movies_data['genres'] = movies_data['genres'].apply(extract_genres)

# 进行独热编码
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movies_data['genres']), columns=mlb.classes_)

# 合并到原数据集
movies_data = pd.concat([movies_data, genres_encoded], axis=1)
movies_data.drop(columns=['genres'], inplace=True)


# 提取前 10 个最常见的制作公司

def extract_companies(company_str):
    try:
        companies = ast.literal_eval(company_str)
        return [company['name'] for company in companies]
    except:
        return []


movies_data['production_companies'] = movies_data['production_companies'].apply(extract_companies)

# 统计出现次数最多的前10个公司
all_companies = [company for sublist in movies_data['production_companies'] for company in sublist]
top_companies = [company for company, _ in Counter(all_companies).most_common(10)]

# 创建二元特征
for company in top_companies:
    movies_data[company] = movies_data['production_companies'].apply(lambda x: 1 if company in x else 0)

# 删除原始 production_companies 列
movies_data.drop(columns=['production_companies'], inplace=True)

# 计算 ROI，避免除零错误
movies_data['ROI'] = movies_data.apply(
    lambda row: (row['revenue'] - row['budget']) / row['budget'] if row['budget'] > 0 else 0, axis=1)


# 创建时长类别
def categorize_runtime(runtime):
    if runtime < 60:
        return 'Short'
    elif 60 <= runtime < 90:
        return 'Medium'
    elif 90 <= runtime < 120:
        return 'Long'
    else:
        return 'Very Long'


movies_data['runtime_category'] = movies_data['runtime'].apply(categorize_runtime)

# 进行独热编码
movies_data = pd.get_dummies(movies_data, columns=['runtime_category'])

# 处理 'release_date' 列，将其转换为 datetime 类型
movies_metadata['release_date'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce')

# 提取日期特征
movies_metadata['release_year'] = movies_metadata['release_date'].dt.year
movies_metadata['release_month'] = movies_metadata['release_date'].dt.month
movies_metadata['release_day'] = movies_metadata['release_date'].dt.day
movies_metadata['release_weekday'] = movies_metadata['release_date'].dt.weekday

# 删除原始的 'release_date' 列，因为我们已经提取了有用的特征
movies_metadata.drop(columns=['release_date'], inplace=True)

# 处理缺失值，填充 'release_year' 的缺失值
movies_metadata['release_year'].fillna(movies_metadata['release_year'].mode()[0], inplace=True)

# 2. 特征选择与目标变量选择
# 选择与票房相关的特征并创建输入数据 X 和目标变量 y
# 假设我们用 'budget' 和提取的日期特征作为特征，目标是 'revenue'
movies_metadata = movies_metadata[
    ['budget', 'release_year', 'release_month', 'release_day', 'release_weekday', 'revenue']]

# 去除含有缺失值的行
movies_metadata.dropna(subset=['budget', 'revenue'], inplace=True)

# 3. 数据划分 - 划分训练集和测试集
X = movies_metadata[['budget', 'release_year', 'release_month', 'release_day', 'release_weekday']]
y = movies_metadata['revenue']

# 数据集拆分 80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 回归模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # 目标函数：平方误差
    n_estimators=500,  # 500 棵树
    learning_rate=0.05,  # 学习率
    max_depth=6,  # 树的最大深度
    subsample=0.8,  # 训练子样本比例
    colsample_bytree=0.8,  # 每棵树使用的特征比例
    random_state=42
)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测测试集
y_pred = xgb_model.predict(X_test)

# 计算误差指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print("Best parameters:", best_params)

param_dist = {
    'max_depth': np.arange(4, 10, 1),
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'n_estimators': np.arange(100, 600, 100),
    'subsample': np.linspace(0.6, 1.0, 5),
    'colsample_bytree': np.linspace(0.6, 1.0, 5)
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 只搜索 20 组参数
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# 获取最佳参数
best_params_random = random_search.best_params_
print("Best parameters (Random Search):", best_params_random)

# 使用最优参数训练 XGBoost
xgb_final = xgb.XGBRegressor(
    **best_params_random,  # 使用随机搜索的最优参数
    objective='reg:squarederror',
    random_state=42
)

xgb_final.fit(X_train, y_train)

# 预测测试集
y_pred_final = xgb_final.predict(X_test)

# 计算最终误差
mse_final = mean_squared_error(y_test, y_pred_final)
rmse_final = np.sqrt(mse_final)
mae_final = mean_absolute_error(y_test, y_pred_final)
r2_final = r2_score(y_test, y_pred_final)

print(f"Final MSE: {mse_final:.2f}")
print(f"Final RMSE: {rmse_final:.2f}")
print(f"Final MAE: {mae_final:.2f}")
print(f"Final R² Score: {r2_final:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_final, alpha=0.6, color="royalblue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")

# 参考线
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred_final, bins=30, kde=True, color="orange")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.show()

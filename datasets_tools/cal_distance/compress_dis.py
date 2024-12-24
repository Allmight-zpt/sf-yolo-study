import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

THRESHOLD = 0.2

# 1. 读取 CSV 文件
df = pd.read_csv('compression_ratios.csv')
df_new = pd.DataFrame()

# 2. 计算第一列减第二列，第二列减第三列
df_new['dis_source'] = df.iloc[:, 0] - df.iloc[:, 1]
df_new['dis_target'] = df.iloc[:, 1] - df.iloc[:, 2]
df_new['file_paths'] = df['file_paths']

# 3. 对计算结果进行归一化
scaler = MinMaxScaler()  # 创建归一化对象
df_new[['dis_source', 'dis_target']] = scaler.fit_transform(df_new[['dis_source', 'dis_target']])


# 4. 四分类逻辑
def categorize(row):
    if row['dis_source'] < THRESHOLD <= row['dis_target']:
        return 0
    elif row['dis_target'] < THRESHOLD <= row['dis_source']:
        return 1
    elif row['dis_source'] < THRESHOLD and row['dis_target'] < THRESHOLD:
        return 2
    else:
        return 3


# 应用四分类逻辑
df_new['category'] = df_new.apply(categorize, axis=1)

# 5. 保存结果到新的 CSV 文件
df_new.to_csv('compression_dis.csv', index=False)

# 6. 使用 seaborn 绘制直方图
plt.figure(figsize=(10, 6))
sns.histplot(df_new['dis_source'], kde=True, color='blue', label='Dis Source')
sns.histplot(df_new['dis_target'], kde=True, color='red', label='Dis Target')

# 添加标签和标题
plt.title('Histogram of Differences')
plt.xlabel('Normalized Difference')
plt.ylabel('Frequency')
plt.legend()

# 显示图表
plt.show()

# 7. 绘制分类后的数量图
plt.figure(figsize=(8, 6))
sns.countplot(x='category', data=df_new, palette='viridis')

# 添加标签和标题
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')

# 显示图表
plt.show()
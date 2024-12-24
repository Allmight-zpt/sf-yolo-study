import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件
df = pd.read_csv('compression_ratios.csv')
df_new = pd.DataFrame()
# 2. 计算第一列减第二列，第二列减第三列
df_new['dis_source'] = df.iloc[:, 0] - df.iloc[:, 1]
df_new['dis_target'] = df.iloc[:, 1] - df.iloc[:, 2]
df_new['file_paths'] = df['file_paths']

# 3. 保存结果到新的 CSV 文件
df_new.to_csv('compression_dis.csv', index=False)

# 4. 使用 seaborn 绘制直方图
plt.figure(figsize=(10, 6))
sns.histplot(df_new['dis_source'], kde=True, color='blue', label='Dis Source')
sns.histplot(df_new['dis_target'], kde=True, color='red', label='Dis Target')

# 添加标签和标题
plt.title('Histogram of Differences')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.legend()

# 显示图表
plt.show()

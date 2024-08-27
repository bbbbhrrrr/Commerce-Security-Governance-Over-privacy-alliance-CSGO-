import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('leveled_Total.csv')
df2 = pd.read_csv('Credit_score.csv')

# 将leveled_Total.csv的level_Total列的值按照ID对齐，添加到Credit_score.csv中
df2['level_Total'] = df1['level_Total']

# 将对齐后的数据保存到新的CSV文件中
df2.to_csv('leveled_Credit_score.csv', index=False)

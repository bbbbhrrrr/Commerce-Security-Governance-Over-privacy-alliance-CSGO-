import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('level_orders_JD.csv')
df2 = pd.read_csv('level_orders_TB.csv')

# 按照第一列ID对齐数据
merged_df = pd.merge(df1, df2, on='ID')

# level为JD和TB的level的平均值
merged_df['level'] = (merged_df['level_x'] + merged_df['level_y']) / 2


# 将合并后的数据写入新的CSV文件
merged_df.to_csv('merged_file.csv', index=False)
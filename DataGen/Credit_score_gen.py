
import pandas as pd
import random
    
file = 'leveled_Total.csv'

# 读取数据
data = pd.read_csv(file, encoding='utf-8')

# 生成信用分
def Score_gen(row):
    # level = 1,2,3,4,5
    # 总分100，每个等级20分，等级越低，信用越好，对于1，2，3，4，5各个等级在各自区间内随机生成一个信用分

    if row['level_Total'] == 1:
        return round(100 - 20 * (1 - random.random()), 2)
    elif row['level_Total'] == 2:
        return round(80 - 20 * (1 - random.random()), 2)
    elif row['level_Total'] == 3:
        return round(60 - 20 * (1 - random.random()), 2)
    elif row['level_Total'] == 4:
        return round(40 - 20 * (1 - random.random()), 2)
    elif row['level_Total'] == 5:
        return round(20 - 20 * (1 - random.random()), 2)
    else:
        return None

# 应用函数到数据框
data['Credit_Score'] = data.apply(Score_gen, axis=1)

columns_to_drop = [
    'Total_Count_Total',
    'Refund_Only_Count_Total',
    'Rental_Not_Returned_Count_Total',
    'Partial_Payment_After_Receipt_Count_Total',
    'Payment_Without_Delivery_Count_Total',
    'Amount_of_Loss_Total',
    'level_Total'
]

# 删除指定的列
data_to_save = data.drop(columns=columns_to_drop)

# 保存数据集

data_to_save.to_csv('Credit_score.csv', index=False, header=['ID', 'Credit_Score'])
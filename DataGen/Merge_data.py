# 使用utf-8编码打开文件，读取文件内容，逐行读取CSV文件中的数据，对数据进行处理，最后将处理后的数据存储到字典中。

import pandas as pd
from tqdm import tqdm

# 将两个csv合并并保存到第三个csv文件中
def merge_data(file1, file2, file3):
    # 先给文件三写入表头

    header = 'Order_ID,Real_Name_User(ID Card),Seller_Information_(ID Card),Product_Information,Product_Amount,Order_Creation_Time,Payment_Time,Shipping_Time,Receiving_Time,Refund_Time,Return_Time,Payment_Amount,Refund_Amount,Platform_Type'

    with open(file3, 'w', encoding='utf-8') as f:
        f.write(header)

    # 读取文件一并跳过表头
    with open(file1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            with open(file3, 'a',encoding='utf-8') as f:
                f.write(line)
    
    # 读取文件二并跳过表头
    with open(file2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            with open(file3, 'a',encoding='utf-8') as f:
                f.write(line)



# 合并数据

merge_data('orders_JD.csv', 'orders_TB.csv', 'orders_Total.csv')
merge_data('orders_JD_Month.csv', 'orders_TB_Month.csv', 'orders_Total_Month.csv')
merge_data('orders_JD_Half_Year.csv', 'orders_TB_Half_Year.csv', 'orders_Total_Half_Year.csv')



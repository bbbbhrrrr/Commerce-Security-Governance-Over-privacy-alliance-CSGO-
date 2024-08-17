'''
id 行为总次数 仅退款不退货次数 租用不还次数 收货后仅支付订金次数 付款不发货次数


'''
import csv
from collections import defaultdict


def Count(file):
    # 定义结果存储字典
    results = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # 初始化每个ID的统计信息


    # 打开CSV文件
    with open(file, mode='r', encoding='utf-8') as file2count:
        # 创建一个CSV读取器
        csv_reader = csv.reader(file2count)
        
        # 读取表头
        header = next(csv_reader)
        
        # 逐行读取CSV文件中的数据
        for row in csv_reader:
            # print(row)  # 打印每一行的数据
            # print(row[0])  # 打印每一行的第一个元素
            Consumer_ID = row[1]
            Producer_ID = row[2]
            Product_Amount = row[4]
            Return_Time = row[10]
            Payment_Amount = row[11]
            Refund_Amount = row[12]
            Shipping_Time = row[7]
            Payment_Time = row[6]
            Platform_Type = row[13]

            results[Consumer_ID][0] += 1
            results[Producer_ID][0] += 1

            Amount_of_Loss = float(Product_Amount) # 初始化造成损失金额
            results[Consumer_ID][5] += Amount_of_Loss    # 初始化消费者造成损失金额

            if Platform_Type == 'lease_platform':
                if Return_Time == '9999':           # 租用不还
                    results[Consumer_ID][2] += 1
                if Return_Time == '9999' and Refund_Amount == Product_Amount:   # 仅退款不退货
                    results[Consumer_ID][1] += 1
                if Product_Amount > Payment_Amount:     # 收货后仅支付订金
                    results[Consumer_ID][5] += float(Product_Amount) - float(Payment_Amount) # 计算损失金额
                    results[Consumer_ID][3] += 1
                if Payment_Amount != 0 and Shipping_Time == '9999':     # 付款不发货
                    results[Producer_ID][4] += 1
                    results[Consumer_ID][5] -= Amount_of_Loss    # 付款不发货，消费者造成损失金额为0             
                    results[Producer_ID][5] += Amount_of_Loss    # 付款不发货，生产者造成损失金额为Amount_of_Loss
            else:
                if Return_Time == '9999' and Refund_Amount == Product_Amount:   # 仅退款不退货
                    results[Consumer_ID][1] += 1
                if Product_Amount > Payment_Amount:     # 收货后仅支付订金
                    results[Consumer_ID][5] += float(Product_Amount) - float(Payment_Amount)
                    results[Consumer_ID][3] += 1
                if Payment_Amount != 0 and Shipping_Time == '9999':     # 付款不发货
                    results[Producer_ID][4] += 1
                    results[Consumer_ID][5] -= Amount_of_Loss    # 付款不发货，消费者造成损失金额为0             
                    results[Producer_ID][5] += Amount_of_Loss    # 付款不发货，生产者造成损失金额为Amount_of_Loss
            


    result_file_name = 'count_' + file
    # 创建并写入新的CSV文件
    with open(result_file_name, mode='w', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        
        # 写入表头
        csv_writer.writerow(['ID', 'Total_Count', 'Refund_Only_Count', 'Rental_Not_Returned_Count', 'Partial_Payment_After_Receipt_Count', 'Payment_Without_Delivery_Count', 'Amount_of_Loss'])
        
        # 写入每个用户的统计信息
        for user_id, counts in results.items():
            csv_writer.writerow([user_id] + counts)


if __name__ == '__main__':
    file1 = 'orders_JD.csv'
    Count(file1)
    file2 = 'orders_JD_Month.csv'
    Count(file2)
    file3 = 'orders_JD_Half_Year.csv'
    Count(file3)
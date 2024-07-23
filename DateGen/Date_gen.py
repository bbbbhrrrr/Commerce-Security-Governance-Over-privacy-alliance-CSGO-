'''
订单号,实名用户（身份证）,商品信息,商品金额,订单生成时间,付款时间,发货时间,收货时间,退款时间,退货时间,付款金额,退款金额,平台类型

Consumer类

Producer类

GenOrder类


生成范围时间 √

生成卖家id √


'''
import csv
import random
import uuid
import json
from datetime import datetime, timedelta

def str_to_datetime(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

# 随机生成日期
def generate_random_datetime(start_year, end_year, least_day=0):
    # 定义起始和结束日期
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # 计算时间跨度
    delta = end_date - start_date
    
    # 生成随机的时间间隔
    random_days = random.randint(0, delta.days)+least_day
    random_seconds = random.randint(0, 86400 - 1)  
    
    # 计算随机日期和时间
    random_date = start_date + timedelta(days=random_days, seconds=random_seconds)
    
    # 格式化输出
    return random_date.strftime("%Y-%m-%d %H:%M:%S")

# 辅助函数生成随机名字
def generate_random_name():
    first_names = ["Alice", "Bob", "Charlie", "David", "Eva"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
    return random.choice(first_names) + " " + random.choice(last_names)


# 随机生成身份证号
def generate_random_id():
    # 身份证号前6位地区码（省市区）
    areas = [
        "110000", "120000", "130000", "140000", "150000", "210000", "220000", "230000",
        "310000", "320000", "330000", "340000", "350000", "360000", "370000", "410000",
        "420000", "430000", "440000", "450000", "460000", "500000", "510000", "520000",
        "530000", "540000", "610000", "620000", "630000", "640000", "650000"
    ]
    # 随机选择一个地区码
    area_code = random.choice(areas)
    
    # 生成随机出生日期
    start_date = datetime(1950, 1, 1)
    end_date = datetime(2006, 12, 31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    birth_date = random_date.strftime("%Y%m%d")
    
    # 生成随机顺序码
    sequence_code = f"{random.randint(0, 999):03d}"
    
    # 计算校验码
    id_no_without_check = area_code + birth_date + sequence_code
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_digits = "10X98765432"
    
    sum_of_products = sum(int(id_no_without_check[i]) * weights[i] for i in range(17))
    check_digit = check_digits[sum_of_products % 11]
    
    return id_no_without_check + check_digit

# 随机生成商品信息
# 读取商品信息文件
with open('product_info.json', 'r', encoding='utf-8') as f:
    categories = json.load(f)
# 生成商品信息
def generate_random_product_info(categories):

    category = random.choice(list(categories.keys()))
    product_type = random.choice(list(categories[category].keys()))
    brand = random.choice(list(categories[category][product_type].keys()))
    model = random.choice(categories[category][product_type][brand])

    product_info = category+product_type+brand+model

    return product_info

# Consumer类
class Consumer:
    def __init__(self):
        self.buyer_id = generate_random_id()
        self.name = generate_random_name()
        self.id_number = generate_random_id()
        self.order_time = None
        self.receive_order_time = None

    def make_payment(self):
        self.order_time = datetime.now()
        return self.order_time.strftime("%Y-%m-%d %H:%M:%S")

    def receive_order(self):
        self.receive_order_time = generate_random_datetime(self.order_time.year, self.order_time.year + 1, 1)
        return self.receive_order_time

    def request_refund(self):
        # print(self.receive_order_time, type(self.receive_order_time))
        return generate_random_datetime(str_to_datetime(self.receive_order_time).year, str_to_datetime(self.receive_order_time).year + 1, 3)

# Producer类
class Producer:
    def __init__(self):
        self.id_number = generate_random_id()
        self.name = generate_random_name()

    def ship_order(self, payment_time=None):
        return generate_random_datetime(str_to_datetime(payment_time).year, str_to_datetime(payment_time).year + 1, 0)

    def process_refund(self, refund_request_time):
        if refund_request_time == 9999:
            return 9999
        else:
            return generate_random_datetime(refund_request_time.year, refund_request_time.year + 1, 1)

# GenOrder类
class GenOrder:
    def __init__(self, output_file,plantfrom):
        self.output_file = output_file
        self.orders = []
        self.plantform = plantfrom

    def generate_order(self, buyer, seller, order_type='normal'):
        order_id = str(uuid.uuid4())
        product_info = generate_random_product_info(categories)
        product_amount = round(random.uniform(10.0, 1000.0), 2)
        payment_time = buyer.make_payment()
        order_creation_time = payment_time
        shipping_time = seller.ship_order(payment_time)
        receiving_time = buyer.receive_order()
        refund_request_time = 9999  # 9999表示未退款
        return_time = 9999  # 9999表示未退货
        payment_amount = product_amount
        refund_amount = 0.0 # 0表示未退款
        platform_type = self.plantform

        # 根据订单类型生成不良订单
        if order_type == 'refund_no_return':  # 仅退款不退货
            refund_request_time = buyer.request_refund()
            return_time = 9999  # 9999表示未退货
            refund_amount = product_amount
        elif order_type == 'rent_not_return':  # 租用不还
            # product_info = "Rent_Product_" + str(random.randint(1, 100))
            platform_type = "租赁平台"
        elif order_type == 'partial_payment':  # 收货后仅支付订金
            deposit_amount = round(product_amount * 0.5, 2)  # 仅支付50%的订金
            payment_amount = deposit_amount
        elif order_type == 'payment_no_shipment':  # 付款不发货
            shipping_time = 9999  # 9999表示未发货
            receiving_time = 9999  # 9999表示未收货
        else:  # 正常订单或处理退款和退货
            if random.choice([True, False]):  # 随机决定是否退款
                return_time = seller.process_refund(refund_request_time)
                refund_amount = product_amount

        if platform_type == "TB" or platform_type == "JD" or platform_type == "PDD" or platform_type == "XY":
            order = {
                "订单号": order_id,
                "实名用户（身份证）": buyer.id_number,
                "商家信息（身份证）": seller.id_number,
                "商品信息": product_info,
                "商品金额": product_amount,
                "订单生成时间": order_creation_time,
                "付款时间": payment_time,
                "发货时间": shipping_time,
                "收货时间": receiving_time,
                "退款时间": refund_request_time,
                "退货时间": return_time,
                "付款金额": payment_amount,
                "退款金额": refund_amount,
                "平台类型": platform_type
            }
            
            self.orders.append(order)

        elif platform_type == "lease_platform":
            order = {
                "订单号": order_id,
                "实名用户（身份证）": buyer.id_number,
                "商家信息（身份证）": seller.id_number,
                "商品信息": product_info,
                "商品金额": product_amount,
                "订单生成时间": order_creation_time,
                "付款时间": payment_time,
                "发货时间": shipping_time,
                "收货时间": receiving_time,
                "到期时间": refund_request_time,
                "发回时间": return_time,
                "付款金额": payment_amount,
                "退款金额": refund_amount,
                "平台类型": platform_type
            }
            
            self.orders.append(order)

    def write_to_csv(self):
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.orders[0].keys())
            writer.writeheader()
            writer.writerows(self.orders)

# Payment_Plantform类
class Payment_Plantform:
    def __init__(self,output_file,plantform):
        self.output_file = output_file
        self.credits = []
        self.plantform = plantform
    
    # 对Consumer和Producer进行信用评级
    def credit_rating(self,person):
        person.credit = random.randint(0, 100)

        credit = {
            "用户（身份证）": person.id_number,
            "信用评级": person.credit,
            "平台类型": self.plantform
        }
        self.credits.append(credit)

    # 写入CSV文件
    def write_to_csv(self):    
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.credits[0].keys())
            writer.writeheader()
            writer.writerows(self.credits)

# 示例使用

# 随机生成十个Consumer和Producer
consumers = [Consumer() for _ in range(10)]
producers = [Producer() for _ in range(10)]

# 生成订单
order_gen_TB= GenOrder("orders_TB.csv","TB")
order_gen_JD= GenOrder("orders_JD.csv","JD")
order_gen_PDD= GenOrder("orders_PDD.csv","PDD")
order_gen_lease_platform= GenOrder("orders_lease_platform.csv","lease_platform")
order_gen_XY = GenOrder("orders_XY.csv","XY")

# 生成支付平台
payment_plantform_AliPay = Payment_Plantform("payment_plantform_AliPay.csv","AliPay")
payment_plantform_WeChat = Payment_Plantform("payment_plantform_WeChat.csv","WeChat")
payment_plantform_CreditCard = Payment_Plantform("payment_plantform_CreditCard.csv","CreditCard")

# 生成普通订单和不良订单
# 分别生成淘宝、京东、拼多多和租赁平台的订单
for _ in range(50):  # 生成50个普通订单
    order_gen_TB.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='normal')  # 淘宝订单
    order_gen_JD.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='normal')  # 京东订单
    order_gen_PDD.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='normal')  # 拼多多订单
    order_gen_lease_platform.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='normal')  # 租赁平台订单
    order_gen_XY.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='normal')  # XY订单
    

for _ in range(10):  # 生成10个不良订单
    order_gen_TB.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='refund_no_return')  # 仅退款不退货订单
    order_gen_JD.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='refund_no_return')  # 仅退款不退货订单
    order_gen_PDD.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='refund_no_return')  # 仅退款不退货订单
    order_gen_lease_platform.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='rent_not_return')  # 租用不还订单
    order_gen_XY.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='partial_payment')  # 收货后仅支付订金订单
    order_gen_XY.generate_order(consumers[random.randint(0, 9)], producers[random.randint(0, 9)], order_type='payment_no_shipment')  # 付款不发货订单

# 生成信用评级
for consumer in consumers:
    payment_plantform_AliPay.credit_rating(consumer)
    payment_plantform_WeChat.credit_rating(consumer)
    payment_plantform_CreditCard.credit_rating(consumer)

for producer in producers:
    payment_plantform_AliPay.credit_rating(producer)
    payment_plantform_WeChat.credit_rating(producer)
    payment_plantform_CreditCard.credit_rating(producer)

# 写入CSV文件
order_gen_TB.write_to_csv()
order_gen_JD.write_to_csv()
order_gen_PDD.write_to_csv()
order_gen_lease_platform.write_to_csv()
order_gen_XY.write_to_csv()

payment_plantform_AliPay.write_to_csv()
payment_plantform_WeChat.write_to_csv()
payment_plantform_CreditCard.write_to_csv()

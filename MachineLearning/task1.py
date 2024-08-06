import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from datetime import datetime


'''
gzc:
task1 1.0版本
采用的机器学习模式为标准化和随机森林分类器
在测试集上评估模型的性能，输出分类报告
数据集划分为训练集和测试集（80% 训练，20% 测试）
'''

#加载数据
data = pd.read_csv('orders_lease_platform.csv')

#数据预处理
#转换时间数据为距离2000年1月1日的天数
def convert_time_to_days(time_str):
    if str(time_str) == '9999' or pd.isnull(time_str):
        return np.nan
    try:
        return (datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') - datetime(2000, 1, 1)).total_seconds() / (60*60*24)
    except ValueError:
        return np.nan

#需要转换的时间列
time_columns = ['Order_Creation_Time_lease_platform', 'Payment_Time_lease_platform', 'Shipping_Time_lease_platform', 
                'Receiving_Time_lease_platform', 'Refund_Time_lease_platform', 'Return_Time_lease_platform']

#对时间列进行转换
for col in time_columns:
    data[col] = data[col].astype(str).apply(convert_time_to_days)

#填充缺失值
data.fillna(-1, inplace=True)

#特征提取
#删除与预测无关的列
features = data.drop(columns=['Order_ID_lease_platform', 'Real_Name_User(ID Card)', 'Product_Information_lease_platform', 'Platform_Type_lease_platform'])
target = data['Platform_Type_lease_platform']

#编码目标变量
le = LabelEncoder()
target_encoded = le.fit_transform(target)

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

#构建模型
pipeline = Pipeline([
    ('scaler', StandardScaler()),  #标准化操作
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
])

#训练模型
pipeline.fit(X_train, y_train)

#评估模型
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

#模型预测概率
def predict_probabilities(new_features):
    new_features_df = pd.DataFrame([new_features], columns=features.columns)  #创建包含新特征的数据框
    return pipeline.predict_proba(new_features_df)  

#示例输入
#确保示例输入的长度与特征列数一致（保证为10个，可以存在占位值）
new_features = [
    843.52,  #订单金额
    convert_time_to_days('2024-07-28 19:43:01'),  #订单创建时间
    convert_time_to_days('2024-07-28 19:43:01'),  #付时间
    convert_time_to_days('2024-06-19 08:20:55'),  #发货时间
    convert_time_to_days('2024-06-23 15:38:56'),  #收货时间
    convert_time_to_days('9999'),  #退款时间
    convert_time_to_days('9999'),  #退货时间
    843.52,  #订单金额占位
    0.0,  #其他特征占位
    1.0   #额外特征，假设为占位值
]

#预测概率
probabilities = predict_probabilities(new_features)
print("分类概率：", probabilities)

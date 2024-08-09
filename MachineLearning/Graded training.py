import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

#加载数据
data = pd.read_csv('count_orders_JD.csv')

#
#基础恶意行为分数
data['Refund_Only_Score'] = np.log(data['Refund_Only_Count'] + np.exp(data['Total_Count']) + np.exp(1) * np.exp(data['Refund_Only_Count'] / data['Total_Count']))
data['Rental_Not_Returned_Score'] = np.log(50 * data['Rental_Not_Returned_Count'] + np.exp(1) * np.exp(data['Total_Count']) + np.exp(1) * np.exp(data['Rental_Not_Returned_Count'] / data['Total_Count']))
data['Partial_Payment_After_Receipt_Score'] = np.log(data['Partial_Payment_After_Receipt_Count'] + np.exp(1) * np.exp(data['Total_Count']) + np.exp(1) * np.exp(data['Partial_Payment_After_Receipt_Count'] / data['Total_Count']))
data['Payment_Without_Delivery_Score'] = np.log(data['Payment_Without_Delivery_Count'] + np.exp(data['Total_Count']) + 50 * np.exp(1) * np.exp(data['Payment_Without_Delivery_Count'] / data['Total_Count']))

#时间序列特征
data['Total_Count_Rolling'] = data['Total_Count'].rolling(window=3, min_periods=1).mean()
data['Refund_Only_Count_Rolling'] = data['Refund_Only_Count'].rolling(window=3, min_periods=1).mean()

#分级规则手动创建标签
def classify_user(row):
    if row['Total_Count'] > 100 and row['Refund_Only_Count'] / row['Total_Count'] > 0.05:
        return '恶意用户'
    elif row['Total_Count'] > 100 and row['Refund_Only_Count'] / row['Total_Count'] > 0.01:
        return '风险用户'
    elif row['Total_Count'] > 10 and row['Refund_Only_Count'] == 0:
        return '优先用户'
    else:
        return '普通用户'

data['User_Category'] = data.apply(classify_user, axis=1)

features = data[['Refund_Only_Score', 'Rental_Not_Returned_Score', 'Partial_Payment_After_Receipt_Score', 
                'Payment_Without_Delivery_Score', 'Total_Count_Rolling', 'Refund_Only_Count_Rolling']]
labels = data['User_Category']

#数据集划分
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

#使用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#使用管道将预处理和模型结合
pipeline = ImbPipeline(steps=[
    ('scaler', RobustScaler()),  #对数据进行稳健标准化
    ('classifier', RandomForestClassifier(random_state=42))
])

#使用网格搜索进行超参数优化
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train_smote, y_train_smote)

#评估
y_pred = grid_search.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("最佳模型参数：", grid_search.best_params_)

#堆叠模型提高性能
estimators = [
    ('rf', RandomForestClassifier(n_estimators=grid_search.best_params_['classifier__n_estimators'], random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

stacking_clf.fit(X_train_smote, y_train_smote)
y_pred_stack = stacking_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred_stack))
print(classification_report(y_test, y_pred_stack))

#进行预测
new_data = [[0.5, 1.2, 0.7, 0.9, 12, 3]]  #示例
new_data_scaled = grid_search.best_estimator_.named_steps['scaler'].transform(new_data)
predicted_category = stacking_clf.predict(new_data_scaled)
print(f"预测用户类别: {predicted_category}")
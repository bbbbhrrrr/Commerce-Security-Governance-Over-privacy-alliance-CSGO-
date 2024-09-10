import secretflow as sf
import matplotlib.pyplot as plt
import pandas as pd
from secretflow.utils.simulation.datasets import dataset
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.utils.simulation.datasets import load_bank_marketing
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
from secretflow.data.vertical import read_csv
from secretflow.security.privacy import DPStrategy, LabelDP
from secretflow.security.privacy.mechanism.tensorflow import GaussianEmbeddingDP
from secretflow.preprocessing.encoder import OneHotEncoder
import tensorflow as tf
import numpy as np
from secretflow.data.vertical import read_csv

ENDC = '\033[0m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'

def get_data(users, spu, self_party=None):
    """获取数据"""

    key_columns = ['ID']
    label_columns = ['ID']

    # 初始化一个空字典来存储路径
    input_path = {}
    # 接受每个用户的输入
    for user in users:
        if self_party is not None and user != self_party:
            input_path[user] = ''
            continue
        path = input(f"{BLUE}[*] 请输入 {user} 的文件路径: {ENDC}")
        input_path[user] = path

    # input_path = {
    #     alice: '/home/GPH/Documents/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_JD.csv',
    #     bob:  '/home/bbbbhrrrr/CSGO/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_TB.csv',
    #     carol:  '/home/lwzheng/workspace/sf/DataGen/leveled_Credit_score.csv'
    # }

    vdf = read_csv(input_path, spu=spu, keys=key_columns,
                   drop_keys=label_columns, psi_protocl="ECDH_PSI_3PC")

    return vdf


def get_predict_data(users, spu, self_party=None):
    """获取预测数据"""

    # 初始化一个空字典来存储路径

    input_path = {}
    # 接受每个用户的输入
    for user in users:
        if self_party is not None and user != self_party:
            input_path[user] = ''
            continue
        path = input(f"{BLUE}[*] 请输入 {user} 的文件路径: {ENDC}")
        input_path[user] = path

    output_path = {}
    
    for user in users:
        if self_party is not None and user != self_party:
            output_path[user] = ''
            continue
        path = input(f"{BLUE}[*] 请输入 {user} 的输出路径: {ENDC}")
        output_path[user] = path

    # print(f"input_path = {input_path}")
    # print(f"output_path = {output_path}")

    spu.psi_csv(
        ['ID'], input_path, output_path, 'carol', protocol='ECDH_PSI_3PC', precheck_input=False, broadcast_result=False
    )

    spu.psi_csv(
        ['ID'], input_path, output_path, 'bob', protocol='ECDH_PSI_3PC', precheck_input=False, broadcast_result=False
    )

    spu.psi_csv(
        ['ID'], input_path, output_path, 'alice', protocol='ECDH_PSI_3PC', precheck_input=False, broadcast_result=False
    )

    print(f"{GREEN}[✓] 隐私求交数据已保存到 {output_path}{ENDC}")

    vdf2 = read_csv(output_path, spu=spu, keys='ID',
                    drop_keys='ID', psi_protocl="ECDH_PSI_3PC")

    return vdf2, output_path, input_path


def gen_train_data(vdf):
    """生成训练数据"""

    label_JD = vdf["level_JD"]
    label_TB = vdf["level_TB"]
    label = vdf["level_Total"]

    # 删除标签列
    data = vdf.drop(columns=["level_JD", "level_TB", "level_Total"])

    # 对数据进行编码
    encoder = LabelEncoder()
    data['Total_Count_JD'] = encoder.fit_transform(data['Total_Count_JD'])
    data['Total_Count_TB'] = encoder.fit_transform(data['Total_Count_TB'])
    data['Refund_Only_Count_JD'] = encoder.fit_transform(
        data['Refund_Only_Count_JD'])
    data['Refund_Only_Count_TB'] = encoder.fit_transform(
        data['Refund_Only_Count_TB'])
    data['Rental_Not_Returned_Count_JD'] = encoder.fit_transform(
        data['Rental_Not_Returned_Count_JD'])
    data['Rental_Not_Returned_Count_TB'] = encoder.fit_transform(
        data['Rental_Not_Returned_Count_TB'])
    data['Partial_Payment_After_Receipt_Count_JD'] = encoder.fit_transform(
        data['Partial_Payment_After_Receipt_Count_JD'])
    data['Partial_Payment_After_Receipt_Count_TB'] = encoder.fit_transform(
        data['Partial_Payment_After_Receipt_Count_TB'])
    data['Payment_Without_Delivery_Count_JD'] = encoder.fit_transform(
        data['Payment_Without_Delivery_Count_JD'])
    data['Payment_Without_Delivery_Count_TB'] = encoder.fit_transform(
        data['Payment_Without_Delivery_Count_TB'])
    data['Amount_of_Loss_JD'] = encoder.fit_transform(
        data['Amount_of_Loss_JD'])
    data['Amount_of_Loss_TB'] = encoder.fit_transform(
        data['Amount_of_Loss_TB'])
    data['Credit_Score'] = encoder.fit_transform(data['Credit_Score'])

    encoder = OneHotEncoder()
    label_JD = encoder.fit_transform(label_JD)
    label_TB = encoder.fit_transform(label_TB)
    label = encoder.fit_transform(label)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # 划分数据集
    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.85, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.85, random_state=random_state
    )

    return train_data, test_data, train_label, test_label


def man_predict_data(vdf):
    """处理预测数据"""

    # label_JD = vdf["level_JD"]
    # label_TB = vdf["level_TB"]
    # label = vdf["level_Total"]

    # 删除标签列
    # data = vdf.drop(columns=["level_JD", "level_TB", "level_Total"])
    data = vdf
    # 对数据进行编码
    encoder = LabelEncoder()
    data['Total_Count_JD'] = encoder.fit_transform(data['Total_Count_JD'])
    data['Total_Count_TB'] = encoder.fit_transform(data['Total_Count_TB'])
    data['Refund_Only_Count_JD'] = encoder.fit_transform(
        data['Refund_Only_Count_JD'])
    data['Refund_Only_Count_TB'] = encoder.fit_transform(
        data['Refund_Only_Count_TB'])
    data['Rental_Not_Returned_Count_JD'] = encoder.fit_transform(
        data['Rental_Not_Returned_Count_JD'])
    data['Rental_Not_Returned_Count_TB'] = encoder.fit_transform(
        data['Rental_Not_Returned_Count_TB'])
    data['Partial_Payment_After_Receipt_Count_JD'] = encoder.fit_transform(
        data['Partial_Payment_After_Receipt_Count_JD'])
    data['Partial_Payment_After_Receipt_Count_TB'] = encoder.fit_transform(
        data['Partial_Payment_After_Receipt_Count_TB'])
    data['Payment_Without_Delivery_Count_JD'] = encoder.fit_transform(
        data['Payment_Without_Delivery_Count_JD'])
    data['Payment_Without_Delivery_Count_TB'] = encoder.fit_transform(
        data['Payment_Without_Delivery_Count_TB'])
    data['Amount_of_Loss_JD'] = encoder.fit_transform(
        data['Amount_of_Loss_JD'])
    data['Amount_of_Loss_TB'] = encoder.fit_transform(
        data['Amount_of_Loss_TB'])
    data['Credit_Score'] = encoder.fit_transform(data['Credit_Score'])

    # encoder = OneHotEncoder()
    # label_JD = encoder.fit_transform(label_JD)
    # label_TB = encoder.fit_transform(label_TB)
    # label = encoder.fit_transform(label)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    return data


def create_base_model(input_dim, output_dim, name='base_model'):
    """创建基础模型"""
    # Create model
    def create_model():
        from tensorflow import keras
        import keras.layers as layers
        import tensorflow as tf

        model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(100, activation="relu"),
                layers.Dense(output_dim, activation="relu"),
            ]
        )
        # Compile model
        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    """创建融合模型"""
    def create_model():
        from tensorflow import keras
        import keras.layers as layers
        import tensorflow as tf

        # input
        input_layers = []
        for i in range(party_nums):
            input_layers.append(
                keras.Input(
                    input_dim,
                )
            )

        merged_layer = layers.concatenate(input_layers)
        fuse_layer = layers.Dense(64, activation='relu')(merged_layer)
        output = layers.Dense(output_dim, activation='sigmoid')(fuse_layer)

        model = keras.Model(inputs=input_layers, outputs=output)
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model


def training(train_data, train_label, test_data, test_label, users):
    """训练模型"""

    alice = users[0]
    bob = users[1]
    carol = users[2]
    # prepare model
    hidden_size = 64

    model_base_alice = create_base_model(6, hidden_size)
    model_base_bob = create_base_model(6, hidden_size)
    carol_model = create_base_model(1, hidden_size)

    model_base_alice()
    model_base_bob()
    carol_model()

    model_fuse = create_fuse_model(
        input_dim=hidden_size, party_nums=3, output_dim=5)
    model_fuse()

    base_model_dict = {alice: model_base_alice,
                       bob: model_base_bob, carol: carol_model}

    # Define DP operations
    train_batch_size = 1000
    gaussian_embedding_dp = GaussianEmbeddingDP(
        noise_multiplier=0.5,
        l2_norm_clip=1.0,
        batch_size=train_batch_size,
        num_samples=train_data.values.partition_shape()[carol][0],
        is_secure_generator=False,
    )
    label_dp = LabelDP(eps=64.0)
    dp_strategy_carol = DPStrategy(label_dp=label_dp)
    dp_strategy_bob = DPStrategy(embedding_dp=gaussian_embedding_dp)
    dp_strategy_alice = DPStrategy(embedding_dp=gaussian_embedding_dp)
    dp_strategy_dict = {alice: dp_strategy_alice,
                        bob: dp_strategy_bob, carol: dp_strategy_carol}
    dp_spent_step_freq = 10

    sl_model = SLModel(
        base_model_dict=base_model_dict,
        device_y=carol,
        model_fuse=model_fuse,
        dp_strategy_dict=dp_strategy_dict,
    )

    history = sl_model.fit(
        train_data,
        train_label,
        validation_data=(test_data, test_label),
        epochs=50,
        batch_size=train_batch_size,
        shuffle=True,
        verbose=1,
        validation_freq=1,
        dp_spent_step_freq=dp_spent_step_freq,
    )

    # Evaluate the model
    evaluator = sl_model.evaluate(test_data, test_label, batch_size=10)
    print(evaluator)

    return history, sl_model


def level_predict(sl_model, test_data, output_path, self_party):
    """预测"""

    # predict the test data
    y_pred = sl_model.predict(test_data)
    # print(f"type(y_pred) = {type(y_pred)}")

    # print(sf.reveal(y_pred))    

    data = sf.reveal(y_pred)

    # 将预测结果转换为 tensor张量

    # 找到最大行数
    max_rows = max(tensor.shape[0] for tensor in data)

    # 填充或裁剪数据，使其形状一致
    padded_data = []
    for tensor in data:
        if tensor.shape[0] < max_rows:
            # 填充
            padding = np.zeros(
                (max_rows - tensor.shape[0], tensor.shape[1]), dtype=np.float32)
            padded_tensor = np.vstack((tensor, padding))
        else:
            # 裁剪
            padded_tensor = tensor[:max_rows, :]
        padded_data.append(padded_tensor)

    # 将数据转换为TensorFlow张量
    tensor = tf.convert_to_tensor(padded_data, dtype=tf.float32)
    # 将 tensor 转换为5列的形式
    tensor = tf.reshape(tensor, [-1, 5])

    # 找到每行最大值的索引
    max_indices = tf.argmax(tensor, axis=1)
    # 将索引转换为 one-hot 编码
    predicted_one_hot = tf.one_hot(max_indices, depth=tensor.shape[1])

    # 打印预测结果和真实标签，作为对比
    # print(f"predicted_one_hot = {predicted_one_hot}")

    # print(sf.reveal(test_label.partitions[carol].data))

    df = pd.DataFrame(1 + tf.argmax(predicted_one_hot, axis=1))

    # output_file = "Commerce-Security-Governance-Over-privacy-alliance-CSGO/Commerce-Security-Governance-Over-privacy-alliance-CSGO--main/DataGen/result.csv"

    output_file = input(f'{BLUE}[*] 请输入等级预测结果保存路径: {ENDC}')

    df.to_csv(output_file, index=False)

    # 读取 Credit_score_psi.csv 和 result.csv，跳过 result.csv 的第一行
    credit_score_df = pd.read_csv(output_path[self_party])
    result_df = pd.read_csv(output_file, header=None, skiprows=1)

    # 合并数据

    merge_data(credit_score_df, result_df, output_file)

    print(f"{GREEN}[✓] 等级预测结果已保存到： {output_file}{ENDC}")

    return output_file


def merge_data(credit_score_df, result_df, output_file):
    """合并数据"""

    # 找到两者中较短的行数，进行截断
    min_length = min(len(credit_score_df), len(result_df))

    # 如果 Credit_score_psi.csv 更长，进行截断
    credit_score_df = credit_score_df.iloc[:min_length]

    # 如果 result.csv 更长，进行截断
    result_df = result_df.iloc[:min_length]

    # 将 result.csv 中的数值替换到 credit_score_df 的 level 列
    credit_score_df['level'] = result_df[0]

    # 将修改后的数据保存到新的 CSV 文件中，或者覆盖原文件
    credit_score_df.to_csv(output_file, index=False)

    # print(f"已成功更新 level 列，处理后的行数为 {min_length} 行。")


def calculate_transaction_limits(plantform,order_amount_path, output_path,self_party_name):

    if self_party_name == 'carol':
        print(f"{RED}[x] 无交易额度计算数据，跳过计算{ENDC}")
        return
    
    # 读取订单金额数据和评级
    order_amount_df = pd.read_csv(order_amount_path)

    # 合并数据
    # merged_df1 = pd.merge(order_amount_df, on='ID')
    merged_df = order_amount_df

    # 计算加权额度
    # 假设 'Amount_of_Loss_Total' 是订单误差金额列，'Credit_Score' 是信誉分列
    merged_df['Weighted_Amount'] = (merged_df['Amount_of_Loss' + plantform].max() - merged_df['Amount_of_Loss'+ plantform]) * (
        merged_df['level'].max() - merged_df['level'] + 0.5) * (merged_df['level'].max() - merged_df['level'] + 0.5)
    merged_df['Transaction_Limit'] = merged_df.groupby(
        'ID')['Weighted_Amount'].transform('sum')

    # 去除重复的 ID 行，保留每个 ID 的交易额度
    transaction_limits = merged_df[[
        'ID', 'Transaction_Limit']].drop_duplicates()

    transaction_limits.to_csv(output_path, index=False)

    print(f"{GREEN}[✓] 交易额度已保存到 {output_path}{ENDC}")


def show_mode_result(history):
    """显示模型结果"""

    # Plot the change of loss during training
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')

    # Plot the change of accuracy during training
    plt.subplot(1, 3, 2)
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot the Area Under Curve(AUC) of loss during training
    plt.subplot(1, 3, 3)
    plt.plot(history['train_auc_1'])
    plt.plot(history['val_auc_1'])
    plt.title('Model Area Under Curve')
    plt.ylabel('Area Under Curve')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    output_path = input(f"{BLUE}[*] 请输入模型展示图片保存路径：{ENDC}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

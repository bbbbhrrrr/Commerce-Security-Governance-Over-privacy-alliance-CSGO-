# ray start --head --node-ip-address="192.168.114.232" --port="20000" --include-dashboard=False --disable-usage-stats

# 拆分学习，传入 main 中的 spu ，输入数据，进行拆分学习
# 参考 https://www.secretflow.org.cn/zh-CN/docs/secretflow/main/tutorial/Split_Learning_for_bank_marketing

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
import logging
import sys
import spu  
from secretflow.data.vertical import read_csv
import re


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# cluster_config ={
#     'parties': {
#         'alice': {
#             'address': '192.168.114.232:9000',
#             'listen_addr': '0.0.0.0:9000'
#         },
#         'bob': {
#             'address': '192.168.114.139:9001',
#             'listen_addr': '0.0.0.0:9001'
#         },
#         'carol': {
#             'address': '192.168.114.133:9002',
#             'listen_addr': '0.0.0.0:9002'
#         },
#     },
#     'self_party': 'alice'
# }


# # SPU settings
# cluster_def = {
#     'nodes': [
#         {'party': 'alice', 'id': 'local:0', 'address': '192.168.114.232:21001'},
#         {'party': 'bob', 'id': 'local:1', 'address': '192.168.114.139:22001'},
#         {'party': 'carol', 'id': 'local:1', 'address': '192.168.114.133:23001'},
#        ],
#    'runtime_config': {
#        'protocol': spu.spu_pb2.SEMI2K,
#        'field': spu.spu_pb2.FM128,
#     },
# }

# link_desc = {
#     'recv_timeout_ms': 3600000,
# }

def generate_cluster_configs_via_input(link_recv_timeout=3600000):
    # 输入alice, bob, carol的IP、端口、SPU端口信息
    parties_info = {}

    for party in ['alice', 'bob', 'carol']:
        print(f"请输入 {party} 的 IP 地址:")
        ip_address = input(f"{party} IP 地址: ")
        while not is_valid_ip(address):
            print("Invalid IP address")
            address = input(
                f"Enter the address of {party}  (e.g. 192.168.0.1): ")

        print("Valid IP address")

        print(f"请输入 {party} 的端口:")
        port = input(f"{party} 端口: ")

        print(f"请输入 {party} 的 SPU 端口:")
        spu_port = input(f"{party} SPU 端口: ")

        parties_info[party] = {
            'address': f"{ip_address}:{port}",
            'listen_addr': f"0.0.0.0:{port}",
            'spu_address': f"{ip_address}:{spu_port}"
        }

    # 输入self_party
    self_party = input("请输入当前节点的身份 (alice/bob/carol): ")

    # 生成 cluster_config
    cluster_config = {
        'parties': {},
        'self_party': self_party
    }

    for party, addr in parties_info.items():
        cluster_config['parties'][party] = {
            'address': addr['address'],
            'listen_addr': addr['listen_addr']
        }

    # 生成 cluster_def
    cluster_def = {
        'nodes': [],
        'runtime_config': {
            'protocol': spu.spu_pb2.SEMI2K,
            'field': spu.spu_pb2.FM128,
        }
    }

    party_ids = ['local:0', 'local:1', 'local:2']

    for i, (party, addr) in enumerate(parties_info.items()):
        cluster_def['nodes'].append({
            'party': party,
            'id': party_ids[i],
            'address': addr['spu_address']
        })

    # 生成 link_desc
    link_desc = {
        'recv_timeout_ms': link_recv_timeout
    }

    return cluster_config, cluster_def, link_desc


# 调用函数生成配置
cluster_config, cluster_def, link_desc = generate_cluster_configs_via_input()


def is_valid_ip(address):
    pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    if re.match(pattern, address):
        return True
    else:
        return False


def is_valid_ip_port(address):
    pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+$'
    if re.match(pattern, address):
        return True
    else:
        return False


ray_address = input(
    "Enter the address of the Ray head node (e.g. 192.168.0.1:9394): ")

while not is_valid_ip(ray_address):
    print("Invalid IP address")
    ray_address = input(
        "Enter the address of the Ray head node (e.g. 192.168.0.1:9394): ")

print("Valid IP address")


# sf init
# <<< !!! >>> replace <192.168.0.1:9394> to your ray head
sf.init(address=ray_address, log_to_driver=True, cluster_config=cluster_config)


alice = sf.PYU('alice')
bob = sf.PYU('bob')
carol = sf.PYU('carol')

users = [alice, bob, carol]

def gen_data(users):

    # init log
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # <<< !!! >>> replace path to real parties local file path.
    key_columns = ['ID']
    label_columns = ['ID']
    spu = sf.SPU(cluster_def, link_desc)

    # 初始化一个空字典来存储路径
    input_path = {}

    # 预定义的用户

    # 接受每个用户的输入
    for user in users:
        path = input(f"请输入 {user} 的文件路径: ")
        input_path[user] = path

        
    # input_path = {
    #     alice: '/home/GPH/Documents/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_JD.csv',
    #     bob:  '/home/bbbbhrrrr/CSGO/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_TB.csv',
    #     carol:  '/home/lwzheng/workspace/sf/DataGen/leveled_Credit_score.csv'
    # }

    vdf = read_csv(input_path, spu=spu, keys=key_columns,
                   drop_keys=label_columns, psi_protocl="ECDH_PSI_3PC")

    label_JD = vdf["level_JD"]
    label_TB = vdf["level_TB"]
    label = vdf["level_Total"]

    data = vdf.drop(columns=["level_JD", "level_TB", "level_Total"])

    # print(vdf.columns)

    # print(data.columns)

    # print(f"label_JD = {type(label_JD)},\n label_TB= {type(label_TB)},\n data= {type(data)}")

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

    random_state = 1234
    train_data, test_data = train_test_split(
        data, train_size=0.85, random_state=random_state
    )
    train_label, test_label = train_test_split(
        label, train_size=0.85, random_state=random_state
    )

    return train_data, test_data, train_label, test_label
# data段


train_data, test_data, train_label, test_label = gen_data(sf)


def calculate_transaction_limits(order_amount_path, credit_score_path, output_path):
    
    # 读取订单金额数据
    order_amount_df = pd.read_csv(order_amount_path)
    
    # 读取信誉分数据
    credit_score_df = pd.read_csv(credit_score_path)
    
    # 合并数据
    merged_df = pd.merge(order_amount_df, credit_score_df, on='ID')
    
    # 计算加权额度
    # 假设 'Amount_of_Loss_Total' 是订单误差金额列，'Credit_Score' 是信誉分列
    merged_df['Weighted_Amount'] = (merged_df['Amount_of_Loss_Total'].max() - merged_df['Amount_of_Loss_Total']) * (merged_df['Credit_Score'] * merged_df['Credit_Score'] / merged_df['Credit_Score'].max())
    merged_df['Transaction_Limit'] = merged_df.groupby('ID')['Weighted_Amount'].transform('sum')
    
    #去除重复的 ID 行，保留每个 ID 的交易额度
    transaction_limits = merged_df[['ID', 'Transaction_Limit']].drop_duplicates()
    
    transaction_limits.to_csv(output_path, index=False)
    
    print(f"交易额度已保存到 {output_path}")



def create_base_model(input_dim, output_dim, name='base_model'):
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


def training(train_data, train_label, test_data, test_label):

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

    # predict the test data
    y_pred = sl_model.predict(test_data)
    print(f"type(y_pred) = {type(y_pred)}")

    print(sf.reveal(y_pred))

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
    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    # 将 tensor 转换为5列的形式
    tensor = tf.reshape(tensor, [-1, 5])

    # 找到每行最大值的索引
    max_indices = tf.argmax(tensor, axis=1)
    # 将索引转换为 one-hot 编码
    predicted_one_hot = tf.one_hot(max_indices, depth=tensor.shape[1])

    # 打印预测结果和真实标签，作为对比
    print(f"predicted_one_hot = {predicted_one_hot}")        

    print(sf.reveal(test_label.partitions[carol].data))

    # Evaluate the model
    evaluator = sl_model.evaluate(test_data, test_label, batch_size=10)
    print(evaluator)


    # 调用函数
    order_amount_path = '/home/GPH/Documents/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_Total.csv'        
    credit_score_path = '/home/GPH/Documents/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_Credit_score.csv'
    output_path = '/home/GPH/Documents/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/transaction_limits.csv'

    calculate_transaction_limits(
        order_amount_path, credit_score_path, output_path)

    return history


history = training(train_data, train_label, test_data, test_label)


def show_mode_loss(history):
    # Plot the change of loss during training
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


def show_mode_accuracy(history):
    # Plot the change of accuracy during training
    plt.plot(history['train_accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def show_mode_auc(history):
    # Plot the Area Under Curve(AUC) of loss during training
    plt.plot(history['train_auc_1'])
    plt.plot(history['val_auc_1'])
    plt.title('Model Area Under Curve')
    plt.ylabel('Area Under Curve')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

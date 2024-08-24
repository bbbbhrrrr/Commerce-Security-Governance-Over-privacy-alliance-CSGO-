import secretflow as sf
import pandas as pd
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder

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

# 初始化
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')

path_dict = {
    alice: '/home/bbbbhrrrr/CSGO/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_JD.csv',
    bob: '/home/bbbbhrrrr/CSGO/Commerce-Security-Governance-Over-privacy-alliance-CSGO-/DataGen/leveled_orders_TB.csv'
}

# Prepare the SPU device
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

vdf = read_csv(path_dict, spu=spu, keys='ID', drop_keys="ID")


print(vdf)

label_JD_vd = vdf["level_JD"]

label_JD = vdf["level_JD"]
label_TB = vdf["level_TB"]


data = vdf.drop(columns=["level_JD", "level_TB"])

print(vdf.columns)

print(data.columns)

print(f"label_JD = {type(label_JD)},\n label_TB= {type(label_TB)},\n data= {type(data)}")


encoder = LabelEncoder()
data['Total_Count_JD'] = encoder.fit_transform(data['Total_Count_JD'])
data['Total_Count_TB'] = encoder.fit_transform(data['Total_Count_TB'])
data['Refund_Only_Count_JD']= encoder.fit_transform(data['Refund_Only_Count_JD'])
data['Refund_Only_Count_TB']= encoder.fit_transform(data['Refund_Only_Count_TB'])
data['Rental_Not_Returned_Count_JD']= encoder.fit_transform(data['Rental_Not_Returned_Count_JD'])
data['Rental_Not_Returned_Count_TB']= encoder.fit_transform(data['Rental_Not_Returned_Count_TB'])
data['Partial_Payment_After_Receipt_Count_JD']= encoder.fit_transform(data['Partial_Payment_After_Receipt_Count_JD'])
data['Partial_Payment_After_Receipt_Count_TB']= encoder.fit_transform(data['Partial_Payment_After_Receipt_Count_TB'])
data['Payment_Without_Delivery_Count_JD']= encoder.fit_transform(data['Payment_Without_Delivery_Count_JD'])
data['Payment_Without_Delivery_Count_TB']= encoder.fit_transform(data['Payment_Without_Delivery_Count_TB'])
data['Amount_of_Loss_JD']= encoder.fit_transform(data['Amount_of_Loss_JD'])
data['Amount_of_Loss_TB']= encoder.fit_transform(data['Amount_of_Loss_TB'])

encoder = OneHotEncoder()
label_JD = encoder.fit_transform(label_JD)
label_TB = encoder.fit_transform(label_TB)

print(f"label_JD = {type(label_JD)},\n label_TB= {type(label_TB)},\n data= {type(data)}")



scaler = MinMaxScaler()
data = scaler.fit_transform(data)

print("===============this is data=====================")
print(data)
print("================this is data====================")


random_state = 1234
train_data, test_data = train_test_split(
    data, train_size=0.85, random_state=random_state
)
train_label, test_label = train_test_split(
    label_JD, train_size=0.85, random_state=random_state
)


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
            optimizer='adam',
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
            optimizer='adam',
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        return model

    return create_model

# prepare model
hidden_size = 64

model_base_alice = create_base_model(6, hidden_size)
model_base_bob = create_base_model(6, hidden_size)

model_base_alice()
model_base_bob()


model_fuse = create_fuse_model(input_dim=hidden_size, party_nums=2, output_dim=5)

model_fuse()

base_model_dict = {alice: model_base_alice, bob: model_base_bob}


# Define DP operations
train_batch_size = 1000
gaussian_embedding_dp = GaussianEmbeddingDP(
    noise_multiplier=0.5,
    l2_norm_clip=1.0,
    batch_size=train_batch_size,
    num_samples=train_data.values.partition_shape()[alice][0],
    is_secure_generator=False,
)
label_dp = LabelDP(eps=64.0)
dp_strategy_alice = DPStrategy(label_dp=label_dp)
dp_strategy_bob = DPStrategy(embedding_dp=gaussian_embedding_dp)
dp_strategy_dict = {alice: dp_strategy_alice, bob: dp_strategy_bob}
dp_spent_step_freq = 10

sl_model = SLModel(
    base_model_dict=base_model_dict,
    device_y=alice,
    model_fuse=model_fuse,
    dp_strategy_dict=dp_strategy_dict,
)

sf.reveal(test_data.partitions[alice].data), sf.reveal(
    test_label.partitions[alice].data
)

sf.reveal(train_data.partitions[alice].data), sf.reveal(
    train_label.partitions[alice].data
)

history = sl_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    epochs=10,
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
        padding = np.zeros((max_rows - tensor.shape[0], tensor.shape[1]), dtype=np.float32)
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

print(sf.reveal(test_label.partitions[alice].data))

# Evaluate the model
evaluator = sl_model.evaluate(test_data, test_label, batch_size=10)
print(evaluator)



# Plot the change of loss during training
plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Plot the change of accuracy during training
plt.plot(history['train_accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot the Area Under Curve(AUC) of loss during training
plt.plot(history['train_auc_1'])
plt.plot(history['val_auc_1'])
plt.title('Model Area Under Curve')
plt.ylabel('Area Under Curve')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()





import secretflow as sf
import pandas as pd
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from secretflow.data.vertical import VDataFrame
import logging
import os
import spu

def init_debug(parties: list):
    sf.init(parties=parties, address="local")
    spu = sf.SPU(sf.utils.testing.cluster_def(parties=parties))
    return spu

def init_prod(ip:str, port:int, cluster_def:dict, parties:list):
    sf.init(parties=parties, address=f'{ip}:{port}')
    spu = sf.SPU(cluster_def=cluster_def)
    return spu

#加上日志文件，用于记录
def setup_logging(log_file="training.log"):
    #设置日志记录
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def create_base_model(input_dim, output_dim, name="base"):
    #创建基础模型
    def create_model():
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(output_dim, activation='relu')(x)
        model = Model(inputs, outputs, name=name)
        return model
    return create_model

def create_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    #创建融合模型
    def create_model():
        inputs = [Input(shape=(input_dim,)) for _ in range(party_nums)]
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(output_dim, activation='sigmoid')(x)
        model = Model(inputs, outputs, name=name)
        return model
    return create_model

def preprocess_data(df: pd.DataFrame, features: list, label: str, parties: list):
    #数据预处理：缩放和编码()

    #将DataFrame转换为VDataFrame
    data_dict = {party: df[features].copy() for party in parties}
    vdf = VDataFrame(data_dict)
    
    #对每个参与方的数据进行缩放处理
    for party in parties:
        party_vdf = vdf[party]
        scaler = MinMaxScaler()
        vdf[party] = scaler.fit_transform(party_vdf)
    
    #处理标签数据
    label_encoder = LabelEncoder()
    df[label] = label_encoder.fit_transform(df[label])

    return vdf, df

def build_vertical_data_frame(df: pd.DataFrame, parties: list, feature_cols: list):
    #构建垂直联邦数据表
    data_dict = {party: df[feature_cols].copy() for party in parties}
    return VDataFrame(data_dict)

def train(parties: list, spu: sf.SPU, df: pd.DataFrame, feature_cols: list, label_col: str):
    #训练模型
    try:
        members = [sf.PYU(party) for party in parties]
        logging.info("初始化参与方完成")

        #数据预处理
        vdf, df = preprocess_data(df, feature_cols, label_col, parties)
        logging.info("数据预处理完成")

        #构建垂直联邦数据表
        train_vdf, test_vdf = train_test_split(vdf, test_size=0.2, random_state=42)
        logging.info("垂直联邦数据表构建完成")

        #创建模型
        base_models = [create_base_model(input_dim=len(feature_cols), output_dim=32, name=f"base_{party}")() for party in parties]
        fuse_model = create_fuse_model(input_dim=32, output_dim=1, party_nums=len(parties), name="fuse_model")()
        sl_model = SLModel(base_models=base_models, fuse_model=fuse_model)
        logging.info("模型创建完成")

        #编译模型
        sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("模型编译完成")

        #将标签转换为 VDataFrame 格式
        label_data = df[[label_col]].copy()
        label_vdf = VDataFrame({party: label_data for party in parties})

        #训练模型
        history = sl_model.fit(
            x=train_vdf,
            y=label_vdf,
            epochs=10,
            batch_size=32,
            validation_data=(test_vdf, label_vdf)
        )
        logging.info("模型训练完成")

        #保存模型
        model_path = 'sl_model.h5'
        sl_model.save(model_path)
        logging.info(f"模型已保存至 {model_path}")

        return sl_model

    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        raise


#运行试试
if __name__ == "__main__":
    #设置日志
    setup_logging()

    #配置参与方和SPU
    #各个参与方需要去共享数据不同的特征（各平台应用）数据集的不同列分配给 parties 列表
    parties = [' ', ]  #示例参与方(需要进一步明确数据格式)
    spu = init_debug(parties)  # 或者使用生产模式的init_prod函数

    #读取数据并定义特征列和标签列
    df = pd.read_csv('count_orders_JD.csv')  #数据集
    feature_cols = ['Total_Count', 'Refund_Only_Count', 'Rental_Not_Returned_Count',
                    'Partial_Payment_After_Receipt_Count', 'Payment_Without_Delivery_Count', 'Amount_of_Loss']
    #需要在数据源更新更新的一列用于最后的评判输出，用户信誉情况评分原始情况吧
    label_col = ' '  #可以根据实际情况选择分类标签(结合用户信誉情况评分？)

    # 训练模型
    sl_model = train(parties, spu, df, feature_cols, label_col)

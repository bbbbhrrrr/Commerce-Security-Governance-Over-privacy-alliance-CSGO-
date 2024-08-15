import secretflow as sf
import pandas as pd
from secretflow.data.split import train_test_split
from secretflow.ml.nn import SLModel
from secretflow.preprocessing.scaler import MinMaxScaler
from secretflow.preprocessing.encoder import LabelEncoder

# 拆分学习，传入 main 中的 spu ，输入数据，进行拆分学习
# 参考 https://www.secretflow.org.cn/zh-CN/docs/secretflow/main/tutorial/Split_Learning_for_bank_marketing

def create_base_model(input_dim, output_dim, name="base"):
    def create_model():
        # 创建基础模型
        pass
    return create_model

def crate_fuse_model(input_dim, output_dim, party_nums, name='fuse_model'):
    def create_model():
        # 创建 fuse 模型
        pass
    return create_model

def train(parties:list, spu:sf.SPU):
    members = []
    for party in parties:
        members.append(sf.PYU(party))
    
    df = pd.read_csv('data.csv')

    # 处理数据，构建垂直联邦表

    # 创建拆分学习模型

    sl_model = SLModel()

    return sl_model

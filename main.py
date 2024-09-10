import argparse
import secretflow as sf
from init import welcome, env_check, skip_check, get_config_triplets
from train import get_data, gen_train_data, training, show_mode_result, get_predict_data, man_predict_data, level_predict, calculate_transaction_limits

ENDC = '\033[0m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'



def work(users, spu, self_party=None, self_party_name=None):

    print(f"{BLUE}[*] 开始收集数据……{ENDC}")

    vdf = get_data(users, spu, self_party)

    print(f"{GREEN}[✓] 数据收集完成: {vdf}{ENDC}")

    print(f"{BLUE}[*] 开始生成训练数据……{ENDC}")

    train_data, test_data, train_label, test_label = gen_train_data(vdf)

    print(f"{GREEN}[✓] 训练数据生成完成: {train_data, test_data, train_label, test_label}{ENDC}")

    print(f"{BLUE}[*] 开始训练模型……{ENDC}")

    history, sl_model= training(train_data, train_label, test_data, test_label, users)

    print(f"{GREEN}[✓] 训练完成: {history}{ENDC}")

    print(f"{BLUE}[*] 开始读取预测数据……{ENDC}")

    vdf2 , output_path, input_path= get_predict_data(users, spu, self_party)

    print(f"{GREEN}[✓] 预测数据读取完成: {vdf2}{ENDC}")

    print(f"{BLUE}[*] 开始处理预测数据……{ENDC}")

    data_pri = man_predict_data(vdf2)

    print(f"{GREEN}[✓] 预测数据处理完成: {data_pri}{ENDC}")

    print(f"{BLUE}[*] 开始预测……{ENDC}")
    
    if self_party is None:  # 本地调试模式下，self_party 设为 alice
        self_party = users[0]
        self_party_name = 'alice'

    output_file = level_predict(sl_model, data_pri, output_path, self_party)

    print(f"{BLUE}[*] 开始计算用户额度限制……{ENDC}")

    plantform = '_' + input_path[self_party].split('/')[-1].split('_')[-1].split('.')[0]

    result_path = input(f"{BLUE}[*] 请输入额度限制结果文件路径：{ENDC}")

    calculate_transaction_limits(plantform,output_file,result_path,self_party_name)

    print(f"{BLUE}[*] 训练结果展示：{ENDC}")
    show_mode_result(history)

    print(f"{GREEN}[✓] 所有任务完成，程序正常退出{ENDC}")

    exit(0)


def main(args):

    if args.debug:

        print(f"{RED}[*] 正在本地调试模式下运行……{ENDC}")

        n = int(input(f"{BLUE}[*] 请输入参与方数量: {ENDC}"))
        if n != 3:
            raise Exception("调试模式下参与方数量必须为 3")
        
        print(f"{BLUE}[*] 正在初始化本地调试 Secretflow 环境……{ENDC}")
        sf.init(['alice', 'bob', 'carol'], address='local')

        print(f"{BLUE}[*] 正在初始化本地调试 SPU 环境……{ENDC}")
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

        print(f"{BLUE}[*] 正在初始化本地调试 PYU……{ENDC}")
        alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')
        users = [alice, bob, carol]

        print(f"{GREEN}[✓] 调试环境初始化完成{ENDC}")

        work(users, spu)

    else:
        cluster_config, cluster_def, link_desc, self_party_name = get_config_triplets(args)

        print(f"{GREEN}[✓] 读取的 cluster_config: {cluster_config}{ENDC}")
        print(f"{GREEN}[✓] cluster_def: {cluster_def}{ENDC}")

        if args.ray_address is None:
            ray_address = input("[*] 请输入本节点 Ray 集群的 URL, 格式为 IP:PORT :")
        else:
            ray_address = args.ray_address[0]

        print(f"{BLUE}[*] 正在初始化 Secretflow 环境……{ENDC}")
        sf.init(address=ray_address, log_to_driver=True,
                cluster_config=cluster_config)

        print(f"{BLUE}[*] 正在初始化 SPU 环境……{ENDC}")
        spu = sf.SPU(cluster_def, link_desc)

        print(f"{BLUE}[*] 正在初始化 PYU ……{ENDC}")
        partis = cluster_config['parties'].keys()  # 仍然是 dict_keys
        users = [f'party_{i+1}' for i in range(len(partis))]
        for i, key in enumerate(partis):  # 直接使用 partis，无需再调用 .keys()
            users[i] = sf.PYU(key)  # 使用 dict 的键而不是通过下标访问
            if key == self_party_name:
                self_party = users[i]
        
        print(f"{GREEN}[✓] 生产环境初始化完成{ENDC}")

        work(users, spu, self_party, self_party_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", nargs=1, metavar="IP:PORT", default=None, type=str, help="指定 Ray 集群的地址")

    parser.add_argument("-c", nargs=1, metavar="config.py",
                        default="config.py", type=str, help="自定义 config.py 路径")
    parser.add_argument("--no-check", action="store_true",  help="跳过环境检测")

    parser.add_argument("--debug", action="store_true", help="开启调试模式")

    args = parser.parse_args()

    welcome()
    if not args.no_check:
        env_check()
    else:
        skip_check()
    
    main(args)

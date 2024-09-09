import argparse
import secretflow as sf
from init import welcome, env_check, skip_check, get_config_triplets
from train import get_data, gen_train_data, training, show_mode_result

def work(users, spu):
    print("[*] 开始收集数据……")

    vdf = get_data(users, spu)

    print(f"[✓] 数据收集完成: {vdf}")

    print("[*] 开始生成训练数据……")

    train_data, test_data, train_label, test_label = gen_train_data(vdf)

    print(f"[✓] 训练数据生成完成: {train_data, test_data, train_label, test_label}")

    print("[*] 开始训练模型……")

    history = training(train_data, train_label, test_data, test_label, users)

    print(f"[✓] 训练完成: {history}")

    # print("[*] 开始计算额度限制……")1

    # print("[*] 开始保存模型……")

    # model_path = input("[*] 请输入模型保存路径: ")

    # users[0].save_model(model_path)

    # print(f"[✓] 模型保存完成: {model_path}")

    print("[*] 训练结果展示：")
    show_mode_result(history)

    print("[✓] 所有任务完成，程序正常退出")

    exit(0)


def main(args):

    if args.debug:

        print("[*] 正在本地调试模式下运行……")

        n = int(input("[*] 请输入参与方数量: "))
        if n != 3:
            raise Exception("调试模式下参与方数量必须为 3")
        
        print("[*] 正在初始化本地调试 Secretflow 环境……")
        sf.init(['alice', 'bob', 'carol'], address='local')

        print("[*] 正在初始化本地调试 SPU 环境……")
        spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

        print("[*] 正在初始化本地调试 PYU ……")
        alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')
        users = [alice, bob, carol]

        print("[✓] 调试环境初始化完成")

        work(users, spu)

    else:
        cluster_config, cluster_def, link_desc = get_config_triplets(args)

        print(f"[✓] 读取的 cluster_config: {cluster_config}")
        print(f"[✓] cluster_def: {cluster_def}")

        ray_address = input("[*] 请输入本节点 Ray 集群的 URL, 格式为 IP:PORT :")

        print("[*] 正在初始化 Secretflow 环境……")
        sf.init(address=ray_address, log_to_driver=True,
                cluster_config=cluster_config)

        print("[*] 正在初始化 SPU 环境……")
        spu = sf.SPU(cluster_def, link_desc)

        print("[*] 正在初始化 PYU……")
        partis = cluster_config['parties'].keys()  # 仍然是 dict_keys
        users = [f'party_{i+1}' for i in range(len(partis))]
        for i, key in enumerate(partis):  # 直接使用 partis，无需再调用 .keys()
            users[i] = sf.PYU(key)  # 使用 dict 的键而不是通过下标访问
        
        print("[✓] 生产环境初始化完成")
        
        work(users, spu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

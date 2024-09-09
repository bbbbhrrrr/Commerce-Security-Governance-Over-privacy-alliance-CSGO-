import argparse
import secretflow as sf
from init import welcome, env_check, skip_check, get_config_triplets
from train import get_data, gen_train_data, training, show_mode_result


def main(args):
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
    partis = cluster_config['parties'].keys()
    users = [f'party_{i+1}' for i in range(len(partis))]
    for i in len(partis):
        users[i] = sf.PYU(partis[i])

    print("[✓] 初始化完成")

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

    print("[✓] 所有任务完成，程序退出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, metavar="config.py",
                        default="config.py", type=str, help="自定义 config.py 路径")
    parser.add_argument("--no-check", action="store_true",  help="跳过环境检测")
    args = parser.parse_args()

    welcome()
    if not args.no_check:
        env_check()
    else:
        skip_check()

    main(args)

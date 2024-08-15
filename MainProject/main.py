import spu
import secretflow as sf

def init_debug(parties:list):
    sf.init(parties=parties, address="local")
    spu = sf.SPU(sf.utils.testing.cluster_def(parties=parties))
    return spu
    
def init_prod(ip:str, port:int, cluster_def:dict, parties:list):
    sf.init(parties=parties, address=f'{ip}:{port}')
    spu = sf.SPU(cluster_def=cluster_def)
    return spu

if __name__ == "__main__":
    n = input("Enter number of parties: ")

    try :
        n = int(n)
    except:
        raise ValueError("Invalid number of parties")
    
    parties = []
    cluster_def = {'nodes': [], 'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL
        }
    }

    # check 
    # https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.8.0b0/getting_started/deployment 
    # for more details about this function

    for i in range(n):
        name = input(f"Enter name of party {i+1}: ")
        parties.append(name)
        cluster_def['nodes'].append({
            'party': name,
            'address': input(f"Enter IP address of party {i+1}: {name}: "),
            'listen_addr': input(f"Enter listen address of party {i+1}: {name}: ")
        })
    
    mode = input("Enter mode (debug/prod): ")
    if mode == "debug":
        spu = init_debug()
    elif mode == "prod":
        ip = input("Enter Head Node IP: ")
        port = int(input("Enter Head Node port: "))
        spu = init_prod(ip, port)
    else:
        raise ValueError("Invalid mode")
    
    print("SPU initialized successfully")

    func = input("Enter function to run: ")

    # 根据输入调用对应的模块中的函数


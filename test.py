import argparse
import json
import math
from typing import List, Any

import numpy as np
from matplotlib import pyplot as plt
import random

import ucb
from server import *
from client import *
import datasets

if __name__ == '__main__':
    w1_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracy_history = {str(w1): [] for w1 in w1_values}
    for w1 in w1_values:
        # 设置命令行程序
        parser = argparse.ArgumentParser(description='Federated Learning')
        parser.add_argument('-c', '--conf', dest='conf')  # argparse默认的变量名是--或-后面的字符串，
        # 但是你也可以通过dest=xxx来设置参数的变量名，然后在代码中用args.xxx来获取参数的值。
        args = parser.parse_args()

        acc_data = []
        loss_data = []
        # acc_data1 = []
        # loss_data1 = []

        # 终端执行python main.py -c ./utils/conf.json。此时，args.conf = ./utils/conf.json
        # 这里我使用文件路径地址替换了"arg.conf"，可以直接运行main.py
        with open('F:\pythonProject\DisasterTask\config.json', 'r') as f:  # args.conf
            conf = json.load(f)

        # 初始化
        N = [0] * 3  # 每个臂的选择次数
        mu = [0] * 3  # 每个臂的平均奖励
        iterations = list(range(0, conf["global_epochs"]))

        train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

        server = Server(conf, eval_datasets)  # Server类实例化。self.eval_loader有 10000/32=313 份测试集batch(32张图片为一份batch)
        clients = []  # 定义客户端列表
        # clients列表里面存放10个客户端类的实例
        for c in range(conf["no_models"]):  # c = 0~9。"c"是客户端的id号
            clients.append(Client(conf, server.global_model, train_datasets, c))
        response_time_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 响应时间
        local_accuracy_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 局部精确度
        select_count_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 选择次数

        candidates = clients  # 首次所有客户端进行本地训练得到各自的初始指标
        print("=====所有客户端各自进行初始本地训练=====")
        # 遍历选中的客户端，每个客户端本地进行训练
        for c in candidates:
            diff, response_time, local_accuracy = c.local_train(server.global_model)
            response_time_set[c.client_id] = response_time
            local_accuracy_set[c.client_id] = local_accuracy * 0.01
            select_count_set[c.client_id] += 1
        print("初始客户端响应时间列表：", response_time_set)
        print("初始客户端局部训练精确度：", local_accuracy_set)
        print("初始客户端选择次数：", select_count_set)
        # print("初始客户端能耗列表：", energy_consumption_set)

        print("\n\n")

        R = [0, 0, 0]
        N = [0, 0, 0]
        c = 2  # 探索参数

        for e in range(conf["global_epochs"]):  # e = 0~2
            print("===============================================================================================")
            print(f"当前是第 {e} 次大循环Global Epoch")

            if e == 0:
                candidates = sorted(range(len(response_time_set)), key=lambda k: response_time_set[k])[:5]
                max_index = 0
            elif e == 1:
                candidates = sorted(range(len(local_accuracy_set)), key=lambda k: local_accuracy_set[k],
                                    reverse=True)[:5]
                max_index = 1
            elif e == 2:
                candidates = sorted(range(len(select_count_set)), key=lambda k: select_count_set[k])[:5]
                max_index = 2
            else:
                ucb_values = ucb.get_ucb(R, N, c)
                print(ucb_values)
                max_index = ucb_values.index(max(ucb_values))
                candidates = ucb.select_clients(ucb_values, response_time_set, local_accuracy_set, select_count_set)

            print("selected clients are: ")
            print(candidates)
            for c in candidates:
                client = clients[c]
                print('client_id: ', client.client_id)
                select_count_set[client.client_id] += 1

            # 累加参数变化。把每一个大循环(global_epochs)的差值加起来
            weight_accumulator = {}
            # 初始化空模型参数weight_accumulator
            for name, params in server.global_model.state_dict().items():
                # 这里务必调试理解！！！
                # name分别为: 'conv1.weight'、'bn1.weight'、'bn1.bias'等等(调试看看);
                # params分别为对应各个name的参数值
                weight_accumulator[name] = torch.zeros_like(params)  # 生成一个和参数矩阵大小相同的零矩阵

            max_response_time = 0
            total_local_accuracy = 0
            mean_local_accuracy = 0
            # 遍历选中的客户端，每个客户端本地进行训练
            for c in candidates:
                client = clients[c]
                diff, response_time, local_accuracy = client.local_train(server.global_model)
                response_time_set[client.client_id] = response_time
                local_accuracy_set[client.client_id] = local_accuracy * 0.01

                total_local_accuracy += local_accuracy * 0.01

                if response_time > max_response_time:
                    max_response_time = response_time

                # 根据客户端返回的参数差值字典更新总体权重
                for name, params in server.global_model.state_dict().items():  # ResNet-18有122个层需要更新参数，所以这里执行122次循环(通过调试理解)
                    weight_accumulator[name].add_(diff[name])

            # 模型参数聚合
            server.model_aggregate(weight_accumulator)  # 执行完这行代码后，模型全局参数就更新了
            # 模型评估
            acc, loss = server.model_eval()

            acc_data.append(acc)
            accuracy_history[str(w1)].append(acc)
            loss_data.append(loss)
            mean_local_accuracy = total_local_accuracy / conf["no_models"]

            # 归一化处理
            inverse_response_time = [1 / x for x in response_time_set]
            inverse_response_time_array = np.array(inverse_response_time)
            normalized_inverse_response_time = (inverse_response_time_array - inverse_response_time_array.min()) / \
                                               (inverse_response_time_array.max() - inverse_response_time_array.min())
            max_normalized_inverse_response_time = max(normalized_inverse_response_time)

            R[max_index] = 0.3 * max_normalized_inverse_response_time + 0.7 * mean_local_accuracy
            N[max_index] += 1

            print("Epoch %d, acc: %f, loss: %f" % (e, acc, loss))

        print(iterations)
        print(acc_data)
        print(loss_data)

        # acc_data = [3.4000000000000004, 26.47, 45.04, 51.9, 55.279999999999994, 57.15, 58.78, 60.089999999999996,
        #             60.809999999999995,
        #             62.370000000000005, 62.73, 64.02, 64.53, 65.53, 66.02, 67.30000000000001, 67.7, 67.73, 68.33, 68.94,
        #             69.69, 69.96,
        #             70.06, 70.33, 70.76, 71.36, 71.26, 71.34, 71.95, 72.1, 72.25, 72.57000000000001, 72.91, 73.04,
        #             73.35000000000001,
        #             73.44000000000001, 73.82, 74.35000000000001, 74.58, 74.59, 74.87, 74.7, 75.09, 74.9, 74.97, 74.82, 75.4,
        #             75.63,
        #             75.74, 76.02, 75.99000000000001, 76.1, 76.32, 76.53, 76.44999999999999, 76.44999999999999, 76.59, 76.96,
        #             76.96,
        #             76.8, 77.22, 77.16, 77.24, 77.01, 77.37, 77.48, 77.34, 77.60000000000001, 77.66999999999999, 77.64,
        #             77.7, 77.92,
        #             78.27, 78.32000000000001, 78.16, 77.94, 78.38000000000001, 78.4, 78.32000000000001, 78.58000000000001,
        #             78.82000000000001, 78.85, 78.8, 78.71000000000001, 78.84, 79.01, 78.94, 78.94, 79.09, 78.96, 79.06,
        #             79.01, 79.33,
        #             79.41, 79.28, 79.41, 79.69000000000001, 79.56, 79.67999999999999, 79.58, 79.53, 79.67999999999999,
        #             79.65, 79.91,
        #             79.63, 80.01, 80.19, 80.08, 80.01, 80.28, 80.31, 80.17999999999999, 80.13, 80.32000000000001,
        #             80.30000000000001,
        #             80.24, 80.32000000000001, 80.49, 80.34, 80.61, 80.42, 80.57, 80.41, 80.64, 80.55, 80.52,
        #             80.58999999999999, 80.85,
        #             80.80000000000001, 80.52, 80.67999999999999, 80.76, 80.89, 80.97, 80.73, 81.17, 81.16, 81.13, 81.19,
        #             81.28, 81.39,
        #             81.08, 81.2, 81.28999999999999, 81.17999999999999, 81.34, 81.35, 81.37, 81.42, 81.15]
        # loss_data = [6.020036882781983, 4.108718675613403, 2.6157983337402344, 1.792639119911194, 1.4580353284835816,
        #              1.2891839738845825, 1.2210407127380372, 1.1633889158248902, 1.1339024394989015, 1.0825557847976686,
        #              1.0777699304580688, 1.050566630935669, 1.0287585438728333, 0.9952281530380249, 0.9783014455795288,
        #              0.9456207675933838, 0.9326059372901917, 0.9334848463058472, 0.9120528416633606, 0.895331557750702,
        #              0.8738118161201477, 0.8605567452430725, 0.8529306189537048, 0.8556652837753296, 0.8423865244865417,
        #              0.8257983393669128, 0.8279648350715637, 0.8277282893180847, 0.8007256634712219, 0.8055884863853454,
        #              0.8039171590328217, 0.7873674789905548, 0.7807137738227844, 0.7743521070480347, 0.770021826171875,
        #              0.774435069847107, 0.7562223616600037, 0.7456477800369262, 0.7471972435951233, 0.7469242653369904,
        #              0.7281157984733582, 0.7334419015407563, 0.7310466807365418, 0.7354077640533447, 0.7349899282455444,
        #              0.7461161173820495, 0.7167631786823273, 0.7144834861755371, 0.711516098022461, 0.7070496664047241,
        #              0.703024151134491, 0.696937461566925, 0.6880398950576783, 0.6893906480789185, 0.6867818823814392,
        #              0.6863624541282654, 0.6835874442100525, 0.6681115579605102, 0.6668490751743317, 0.6760427379608154,
        #              0.6668116636753082, 0.6705989500999451, 0.6642472212791443, 0.6683563977241516, 0.6598036423683167,
        #              0.6529325654029846, 0.6521110607147217, 0.6445813700675964, 0.6416202862739563, 0.6515511241436005,
        #              0.6467690301895141, 0.646712167263031, 0.6337352444648743, 0.6319804413318634, 0.6359941535949707,
        #              0.6410130393505097, 0.6273667146682739, 0.6226467041015625, 0.6239295996665954, 0.6213277770042419,
        #              0.6118856323242188, 0.6100673275470734, 0.6113582828998566, 0.6186518151283265, 0.6159101406097413,
        #              0.6087525002002716, 0.6099436645269394, 0.6035080131530761, 0.6001990396976471, 0.6017145503520965,
        #              0.602581467962265, 0.6019061944961548, 0.5989355967521668, 0.5972248303413391, 0.596113382434845,
        #              0.5915794830322265, 0.5880416561603546, 0.5883285219192504, 0.5869388162136078, 0.5888211212158203,
        #              0.59239768242836, 0.588961643075943, 0.5899989702701569, 0.584341903924942, 0.5896306457996369,
        #              0.5799847342967988,
        #              0.5744282806158065, 0.5750227921009063, 0.5784627967357635, 0.5701302674770355, 0.569794760632515,
        #              0.5723394791603088, 0.5743300560951233, 0.570503332567215, 0.5701791202068329, 0.5663796729326248,
        #              0.5662396997451782, 0.5662712559223175, 0.5667485763549804, 0.561197555065155, 0.5630323071479797,
        #              0.5611171702861786, 0.5637984433174134, 0.5641341955661774, 0.5632753230571746, 0.5607921362876892,
        #              0.5587863938331604, 0.55289435505867, 0.5553242135286331, 0.5585266085147857, 0.5563301313877106,
        #              0.5515003259658814, 0.5519712920665741, 0.552530802154541, 0.5530554036617279, 0.5487321749687195,
        #              0.5469820370674133, 0.5489557826042175, 0.550511703968048, 0.5509819007396698, 0.5502182390213013,
        #              0.5519274525165558, 0.5473055596113205, 0.5444628908634186, 0.5474637306213379, 0.5465238233089447,
        #              0.5453432023525238, 0.5454091389656067, 0.5455678564548493, 0.5480009329795837]
        #
        # # =============================随机算法 对比1
        # for e in range(conf["global_epochs"]):
        #     print("Global Epoch %d" % e)
        #     # 每次训练都是从clients列表中随机采样k个进行本轮训练
        #     candidates = random.sample(clients, conf["k"])
        #     print("select clients is: ")
        #     for c in candidates:
        #         print(c.client_id)
        #
        #     # 权重累计
        #     weight_accumulator = {}
        #
        #     # 初始化空模型参数weight_accumulator
        #     for name, params in server.global_model.state_dict().items():
        #         # 生成一个和参数矩阵大小相同的0矩阵
        #         weight_accumulator[name] = torch.zeros_like(params)
        #
        #     # 遍历客户端，每个客户端本地训练模型
        #     for c in candidates:
        #         diff = c.local_train(server.global_model)
        #
        #         # 根据客户端的参数差值字典更新总体权重
        #         for name, params in server.global_model.state_dict().items():
        #             weight_accumulator[name].add_(diff[name])
        #
        #     # 模型参数聚合
        #     server.model_aggregate(weight_accumulator)
        #
        #     # 模型评估
        #     acc, loss = server.model_eval()
        #
        #     acc_data1.append(acc)
        #     loss_data1.append(loss)
        #
        # print(iterations)
        # print(acc_data1)
        # print(loss_data1)
        # # =====================================
        #
        # selected_iterations = iterations[::2]
        # selected_acc_data = acc_data[::2]
        # selected_loss_data = loss_data[::2]
        # selected_acc_data1 = acc_data1[::2]
        # selected_loss_data1 = loss_data1[::2]
        #
        # print("selected_iterations：", selected_iterations)
        # print("selected_acc_data：", selected_acc_data)
        # print("selected_loss_data：", selected_loss_data)
        # print("selected_acc_data1：", selected_acc_data1)
        # print("selected_loss_data1：", selected_loss_data1)
        #
        # # 创建图表
        # plt.figure(figsize=(10, 6))
        # plt.plot(selected_iterations, selected_acc_data, label='proposed MCCS algorithm')
        # plt.plot(selected_iterations, selected_acc_data1, label='random selection method')
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Comparison of Training Accuracy for Two Algorithms')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # # 绘制准确率与迭代次数的关系
        # plt.figure(figsize=(10, 5))  # 创建一个新窗口
        # plt.plot(selected_iterations, selected_acc_data, 'b', label='Accuracy')
        # plt.xlabel('Number of rounds')
        # plt.ylabel('Test accuracy')
        # plt.title('Test accuracy over number of rounds with CIFAR-10 data set.')
        # plt.legend()
        # # 保存图像到本地
        # plt.savefig('E:/桌面/运行日志/accuracy_chart002.png')
        # plt.close()  # 关闭图像窗口
        #
        # # 绘制损失与迭代次数的关系
        # plt.figure(figsize=(10, 5))  # 创建另一个新窗口
        # plt.plot(selected_iterations, selected_loss_data, 'r', label='Loss')
        # plt.xlabel('Number of rounds')
        # plt.ylabel('Training loss')
        # plt.title('Training loss over number of rounds with CIFAR-10 data set.')
        # plt.legend()
        # # 保存图像到本地
        # plt.savefig('E:/桌面/运行日志/loss_chart002.png')
        # plt.close()  # 关闭图像窗口
    for w1, accs in accuracy_history.items():
        plt.plot(accs, label=f'w1={w1}')

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations for Different w1 Values')
    plt.legend()
    plt.savefig('E:/桌面/运行日志/w1 Values.png')
    plt.close()
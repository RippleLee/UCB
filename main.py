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
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')  # argparse默认的变量名是--或-后面的字符串，
    # 但是你也可以通过dest=xxx来设置参数的变量名，然后在代码中用args.xxx来获取参数的值。
    args = parser.parse_args()

    acc_data = []
    loss_data = []
    acc_data1 = []
    loss_data1 = []

    # 终端执行python main.py -c ./utils/conf.json。此时，args.conf = ./utils/conf.json
    # 这里我使用文件路径地址替换了"arg.conf"，可以直接运行main.py
    with open('F:\pythonProject\TEST\config.json', 'r') as f:  # args.conf
        conf = json.load(f)

    # 初始化
    iterations = list(range(0, conf["global_epochs"]))

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)  # Server类实例化。self.eval_loader有 10000/32=313 份测试集batch(32张图片为一份batch)
    clients = []  # 定义客户端列表

    # clients列表里面存放10个客户端类的实例
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))
    # response_time_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 响应时间
    # local_accuracy_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 局部精确度
    # select_count_set = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 选择次数

    # candidates = clients  # 首次所有客户端进行本地训练得到各自的初始指标
    # print("=====所有客户端各自进行初始本地训练=====")
    # # 遍历选中的客户端，每个客户端本地进行训练
    # for c in candidates:
    #     diff, response_time, local_accuracy = c.local_train(server.global_model)
    #     response_time_set[c.client_id] = response_time
    #     local_accuracy_set[c.client_id] = local_accuracy * 0.01
    #     select_count_set[c.client_id] += 1
    # print("初始客户端响应时间列表：", response_time_set)
    # print("初始客户端局部训练精确度：", local_accuracy_set)
    # print("初始客户端选择次数：", select_count_set)
    # # print("初始客户端能耗列表：", energy_consumption_set)

    print("\n\n")

    R = [0]*conf["no_models"]
    N = [0]*conf["no_models"]
    local_accuracy_set=[0]*conf["no_models"]
    local_loss_set=[0]*conf["no_models"]
    total_seconds_set=[0]*conf["no_models"]
    c = 2  # 探索参数
    accuracy_weight = 0.1
    loss_weight = 0.6
    time_weight = 0.3

    for e in range(conf["global_epochs"]):  # e = 0~2
        print("===============================================================================================")
        print(f"当前是第 {e} 次大循环Global Epoch")

        ucb_values = ucb.get_ucb(R, N, c)
        candidates = ucb.select_clients(ucb_values, conf["k"])
        print(f"客户端的ucb值为：{ucb_values}")

        print("selected clients are: ")
        print(candidates)
        for c in candidates:
            client = clients[c]
            print('client_id: ', client.client_id)
            N[client.client_id] += 1

        # 累加参数变化。把每一个大循环(global_epochs)的差值加起来
        weight_accumulator = {}
        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 这里务必调试理解！！！
            # name分别为: 'conv1.weight'、'bn1.weight'、'bn1.bias'等等(调试看看);
            # params分别为对应各个name的参数值
            weight_accumulator[name] = torch.zeros_like(params)  # 生成一个和参数矩阵大小相同的零矩阵

        # max_response_time = 0
        # total_local_accuracy = 0
        # mean_local_accuracy = 0
        # 遍历选中的客户端，每个客户端本地进行训练
        for c in candidates:
            client = clients[c]
            diff, local_accuracy, local_loss, total_seconds = client.local_train(server.global_model)
            local_accuracy_set[client.client_id]=local_accuracy
            local_loss_set[client.client_id] = local_loss
            total_seconds_set[client.client_id] = total_seconds

            # 根据客户端返回的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():  # ResNet-18有122个层需要更新参数，所以这里执行122次循环(通过调试理解)
                weight_accumulator[name].add_(diff[name])

        normalized_accuracy = [(x - min(local_accuracy_set)) / (max(local_accuracy_set)- min(local_accuracy_set)) for x in local_accuracy_set]
        normalized_loss = [(x - min(local_loss_set)) / (max(local_loss_set)- min(local_loss_set)) for x in local_loss_set]
        normalized_time = [(x - min(total_seconds_set)) / (max(total_seconds_set)- min(total_seconds_set)) for x in total_seconds_set]

        for i in range(conf["no_models"]):
            # 注意：损失是越小越好，所以我们用1减去归一化损失
            reward = (accuracy_weight * normalized_accuracy[i] +
                      loss_weight * normalized_loss[i] +
                      time_weight * (1 - normalized_time[i]))
            R.append(reward)

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)  # 执行完这行代码后，模型全局参数就更新了
        # 模型评估
        acc, loss = server.model_eval()
        if acc > 80:
            accuracy_weight = 0.6
            loss_weight = 0.1
        acc_data.append(acc)
        loss_data.append(loss)

        print("Epoch %d, acc: %f, loss: %f" % (e, acc, loss))
        print("客户端响应时间列表：", total_seconds_set)
        print("客户端局部训练精确度：", local_accuracy_set)
        print("客户端局部损失：", local_loss_set)
        print("每个客户端的奖励均值：", R)
        print("每个客户端的选择次数：", N)

    print(iterations)
    print(acc_data)
    print(loss_data)

    # =============================随机算法 对比1
    local_accuracy_set = [0] * conf["no_models"]
    local_loss_set = [0] * conf["no_models"]
    total_seconds_set = [0] * conf["no_models"]
    iterations = list(range(0, conf["global_epochs"]))

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)  # Server类实例化。self.eval_loader有 10000/32=313 份测试集batch(32张图片为一份batch)
    clients = []  # 定义客户端列表
    # clients列表里面存放10个客户端类的实例
    for c in range(conf["no_models"]):  # c = 0~9。"c"是客户端的id号
        clients.append(Client(conf, server.global_model, train_datasets, c))

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样k个进行本轮训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)

        # 权重累计
        weight_accumulator = {}

        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)

        # 遍历客户端，每个客户端本地训练模型
        for c in candidates:
            diff, local_accuracy, local_loss, total_seconds = client.local_train(server.global_model)
            local_accuracy_set[client.client_id] = local_accuracy
            local_loss_set[client.client_id] = local_loss
            total_seconds_set[client.client_id] = total_seconds

            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        # 模型评估
        acc, loss = server.model_eval()

        acc_data1.append(acc)
        loss_data1.append(loss)

        print("Epoch %d, acc: %f, loss: %f" % (e, acc, loss))
        print("客户端响应时间列表：", total_seconds_set)
        print("客户端局部训练精确度：", local_accuracy_set)
        print("客户端局部损失：", local_loss_set)

    print(iterations)
    print(acc_data1)
    print(loss_data1)
    # =====================================

    # selected_iterations = iterations[::2]
    # selected_acc_data = acc_data[::2]
    # selected_loss_data = loss_data[::2]
    # selected_acc_data1 = acc_data1[::2]
    # selected_loss_data1 = loss_data1[::2]
    selected_iterations = iterations[::2]
    selected_acc_data = acc_data[::2]
    selected_loss_data = loss_data[::2]
    selected_acc_data1 = acc_data1[::2]
    selected_loss_data1 = loss_data1[::2]

    print("selected_iterations：", selected_iterations)
    print("selected_acc_data：", selected_acc_data)
    print("selected_loss_data：", selected_loss_data)
    print("selected_acc_data1：", selected_acc_data1)
    print("selected_loss_data1：", selected_loss_data1)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(selected_iterations, selected_acc_data, label='proposed MCCS algorithm')
    plt.plot(selected_iterations, selected_acc_data1, label='random selection method')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Training Accuracy for Two Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('E:/桌面/运行日志/MCCSvsRandom_scc_20241111.png')
    plt.close()

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(selected_iterations, selected_loss_data, label='proposed MCCS algorithm')
    plt.plot(selected_iterations, selected_loss_data1, label='random selection method')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Comparison of Loss for Two Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('E:/桌面/运行日志/MCCSvsRandom_Loss_20241111.png')
    plt.close()

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

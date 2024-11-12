import time
import models, torch, copy
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random



# 客户端类
class Client(object):  # clients.append(Client(conf, server.global_model, train_datasets, c))
    # 构造函数
    def __init__(self, conf, model, train_dataset, id=-1):
        # 读取配置文件
        self.conf = conf
        # 根据配置文件获取客户端本地模型（一般由服务器传输）
        self.local_model = models.get_model(self.conf["model_name"])
        # 客户端ID
        self.client_id = id
        # 客户端本地数据集
        self.train_dataset = train_dataset
        # 获取整个数据集的索引
        all_indices = list(range(len(self.train_dataset)))
        # 随机打乱索引
        random.shuffle(all_indices)
        # 确定每个客户端应该获得的数据量，这里使用随机数来决定
        # 计算每个客户端平均应该分配的数据量
        average_data_size = len(self.train_dataset) // conf["no_models"]

        # 设置数据量的上下限，这里以平均数据量的百分比来定义
        min_percentage = 0.8  # 最少80%的平均数据量
        max_percentage = 1.2  # 最多120%的平均数据量

        # 计算实际的最小和最大数据量
        min_data_size = int(average_data_size * min_percentage)
        max_data_size = int(average_data_size * max_percentage)
        data_size = random.randint(min_data_size, max_data_size)
        # 为当前客户端随机选择数据
        train_indices = random.sample(all_indices, data_size)
        # 创建DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                       sampler=SubsetRandomSampler(train_indices))
        print(f"Client {self.client_id}: Data size = {data_size}")

    # SubsetRandomSampler用来打乱数据
    # 模型本地训练函数
    def local_train(self, model):

        start_time = time.time()
        # 客户端获取服务器的模型，然后通过部分本地数据集进行训练
        for name, param in model.state_dict().items():
            # 用服务器下发的全局模型参数(weight、bias)覆盖本地模型参数(weight、bias)。
            # 本文服务端、客户端采用的都是ResNet-18模型，所以其实参数时一样的，但是按照代码完整性，还是覆盖一下。
            self.local_model.state_dict()[name].copy_(param.clone())

        # print(id(model))
        # 定义最优化函数器用户本地模型训练
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        # print(id(self.local_model))
        # 本地训练模型
        # 设置开启模型训练，如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加local_model.train()，在测试时添加local_model.eval()
        self.local_model.train()
        # 开始训练模型，本文每个客户端都是本地训练3次
        correct = 0
        total = 0
        total_loss = 0
        total_batches = 0
        for e in range(self.conf["local_epochs"]):
            # 每次(共3次)本地训练训练5000张图片，每次下面的循环需要循环157次
            for batch_id, batch in enumerate(self.train_loader):  # self.train_loader有5000张图片，分成了5000/32=157份
                data, target = batch

                if torch.cuda.is_available():  # 如果可以的话加载到gpu
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()  # 在计算参数的梯度之前，通常需要清零梯度信息。清零模型参数的梯度值
                output = self.local_model(data)  # # 训练预测
                loss = torch.nn.functional.cross_entropy(output, target)  # 计算损失函数cross_entropy交叉熵误差
                total_loss += loss.item()  # 累加损失
                total_batches += 1  # 累加批次数
                loss.backward()  # 利用自动求导函数loss.backward()求出模型中所有参数的梯度信息"loss对权重或偏置求导"，这些梯度会自动保存在每个张量的grad成员变量中
                optimizer.step()  # 根据梯度下降算法更新参数。w'=w-lr*grad; b'=b-lr*grad

                # 计算精确度
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print(f'客户端: {self.client_id} 的', "Epoch %d done." % e)
        end_time = time.time()
        total_seconds = end_time - start_time
        local_accuracy = correct / total
        local_loss = total_loss / total_batches
        print(f'客户端: {self.client_id} 本地训练精确度为: {local_accuracy:.2f}，局部损失为：{local_loss:.2f}，共计用时：{total_seconds}s')

        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        # 此时self.local_model.state_dict()和model.state_dict()不一样了
        for name, data in self.local_model.state_dict().items():  # ResNet-18有122个层需要更新参数，所以这里执行122次循环(通过调试理解)
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])

        # 客户端返回差值  本地模型参数与全局模型参数的差值  参数的差值张量
        return diff, local_accuracy, local_loss, total_seconds

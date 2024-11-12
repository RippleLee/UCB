import models, torch


# 服务器类
class Server(object):

    def __init__(self, conf, eval_dataset):  # 定义构造函数

        self.conf = conf  # 将配置信息拷贝到服务端中

        self.global_model = models.get_model(self.conf["model_name"])  # 按照配置中的模型信息获取模型，返回类型为model(nn.Module)

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    # 模型聚合函数
    # weight_accumulator 存储了每个客户端上传参数的变化值
    def model_aggregate(self, weight_accumulator):
        # 遍历服务器的全局模型
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
            # 累加
            if data.type() != update_per_layer.type():
                # 因为update_per_layer的type是floatTensor，所以将其转换为模型的LongTensor（损失精度）
                # 结果为聚合之后的全局模型的weight、bias
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 模型评估函数
    def model_eval(self):
        # 开启模型评估模式
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(self.eval_loader):  # self.eval_loader有 313 份测试集batch(32张图片为一份batch)
            data, target = batch
            # 获取所有样本总量大小
            dataset_size += data.size()[0]  # data.size()=torch.Size([32, 3, 32, 32])

            if torch.cuda.is_available():  # 如果可以的话存储到gpu
                data = data.cuda()
                target = target.cuda()

            # 加载到模型中训练
            output = self.global_model(data)  # output的shape为(32, 1000)
            # 聚合所有损失 cross_entropy 交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            # 获取最大的对数概率的索引值，即在所有预测结果中选择可能性最大的作为最终结果
            # 解读output.data.max(1): 首先output的shape是(32, 1000)。通过output的第二个维度找到每行最大值(共32个)，返回是值和索引(通过调试理解)
            # 解读output.data.max(1)[1]: 提取出"索引"
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            # 统计预测结果与真实标签的匹配个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        # 计算准确率
        acc = 100.0 * (float(correct) / float(dataset_size))
        # 计算总损失值
        total_l = total_loss / dataset_size

        return acc, total_l

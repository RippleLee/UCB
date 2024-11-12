import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Environment(object):

    def __init__(self):
        self.D = None
        self.angle = None
        self.v = None
        self.mec_location = None
        self.uav_location = None
        self.uav_num = 25  # 无人机客户端数量
        self.mec_num = 1  # 服务器数量
        self.f_uav = 1.5 * (10 ** 9)  # 分配给无人机的计算资源
        self.task_num = 4  # 传感任务种类
        self.uav_distance = 400  # uav初始位置之间的间距
        self.mec_location_x = 0
        self.mec_location_y = 0
        self.beta_0 = -30  # 1m参考距离的信道增益
        self.H = 100  # 无人机的飞行高度
        self.B0 = 2*10 ** 5  # 分配给无人机的带宽
        self.sigma2 = 10 ** -5  # 无人机处的噪声功率
        self.P_uav = None  # 无人机的发射功率
        self.C_uav = 1000  # 无人机处理1bit数据需要的CPU计算周期数

        self.FL_L = 4  # 损失函数中的常数
        self.FL_zeta = 2  # 损失函数中的常数
        self.FL_rho = 1/3  # 超学习参数
        self.FL_delta = 1/4  # 超学习参数
        self.FL_eta = None  # 全局精度
        self.FL_lota = 10**(-3)  # 联邦学习的局部精度

    def reset(self):
        self.uav_location = np.empty(shape=(self.uav_num, 2))  # 初始化用户位置数组，用于存储每个无人机的x和y坐标
        self.mec_location = np.empty(shape=(self.mec_num, 2))  # 初始化无人机位置数组，用于存储服务器对应的x和y坐标
        self.v = np.empty(self.uav_num)
        self.angle = np.empty(self.uav_num)
        self.D = np.empty(shape=(self.uav_num, self.task_num))  # 初始化无人机数据量大小数组，用于存储每个无人机对每个任务数据量

        # 初始化无人机的位置
        for i in range(self.uav_num):  # 遍历所有无人机
            uav_location_x = (i % 5) * self.uav_distance
            uav_location_y = (i // 5) * self.uav_distance
            self.uav_location[i][0] = uav_location_x  # 存储无人机x位置
            self.uav_location[i][1] = uav_location_y  # 存储无人机y位置

        # 初始化任务数据量大小
        for i in range(self.uav_num):  # 遍历所有无人机
            for j in range(self.task_num):  # 遍历所有任务
                d_ = random.uniform(1, 3) * (10 ** 6)  # 随机初始化任务数据量大小为1到10兆字节之间
                self.D[i][j] = d_  # 存储用户任务数据量大小

        # 初始化服务器的位置
        self.mec_location[0][0] = self.mec_location_x
        self.mec_location[0][1] = self.mec_location_y

        # 创建无人机模型，包括无人机的位置、移动方向、移动速度和本地计算能力
        uav = np.empty(shape=(self.uav_num, 5))

        # 创建任务模型，包括每个无人机上任务量的大小
        task = np.empty(shape=(self.uav_num, 5))

        for i in range(self.uav_num):  # 遍历所有无人机
            uav[i][0] = self.uav_location[i][0]
            uav[i][1] = self.uav_location[i][1]
            uav[i][2] = self.v[i]
            uav[i][3] = self.angle[i]
            uav[i][4] = self.f_uav
            for j in range(self.task_num):  # 遍历所有任务
                task[i][j] = self.D[i][j]

        return uav, task

    def System_Model(self):
        channel_gain = np.empty(self.uav_num)
        r = np.empty(self.uav_num)
        t_up = np.empty(shape=(self.uav_num, 1))
        t_local = np.empty(shape=(self.uav_num, 1))
        t_total = np.empty(shape=(self.uav_num, 1))
        I = np.empty(shape=(self.uav_num, 1))

        # 通信模型
        # 计算每个无人机与中心节点服务器之间的信道增益
        for i in range(self.uav_num):
            d = (self.uav_location[i][0] - self.mec_location[0][0]) ** 2 + (
                    self.uav_location[i][1] - self.mec_location[0][1]) ** 2
            gain = self.beta_0 / (self.H ** 2 + d)
            channel_gain[i] = gain

        # 计算每个无人机与中心节点服务器之间数据的传输速率
        for i in range(self.uav_num):
            r_ = self.B0 * math.log(1 + self.P_uav * channel_gain[i] / (self.beta_0 * self.sigma2) , 2)
            r[i] = r_

        # 服务延迟
        # 1）本地训练时延
        for i in range(self.uav_num):
            I_ = (2 * math.log(1 / self.FL_eta, 2)) / ((2 - self.FL_L * self.FL_delta) * self.FL_delta * self.FL_zeta)
            I[i] = I_
            t_local_ = 0
            for j in range(self.task_num):
                t_local_ += (I_ * self.C_uav * self.D[i][j]) / self.f_uav
            t_local[i] = t_local_


        # 2）模型传输时延
        for i in range(self.uav_num):
            t_up_ = 0
            for j in range(self.task_num):
                t_up_ += self.D[i][j] / r[i]
            t_up[i] = t_up_

        # 3）全局聚合时延
        # 总时延
        a = ((2 * self.FL_L ** 2) / (self.FL_zeta ** 2 * self.FL_rho)) / math.log(1 / self.FL_lota, math.e)
        I_total = a / (1 - self.FL_eta)
        for i in range(self.uav_num):
            t_total_ = I_total * (t_local[i] + t_up[i])
            t_total[i] = t_total_
        T =(max(t_total)).item()

        return T

    def simulate_delay(self, p_i, eta):
        # 更新传输功率和局部训练精度
        self.P_uav = p_i
        self.FL_eta = eta

        # 重置环境
        uav, task = self.reset()

        # 计算系统模型并返回总时延
        return self.System_Model()


if __name__ == '__main__':
    env = Environment()
    # 定义传输功率和局部训练精度的范围
    p_i_values = np.linspace(30, 50, 5)  # 传输功率范围从 10 到 50 dBm，共 5 个点
    eta_values = np.linspace(0.1, 0.9, 9)  # 局部训练精度范围从 0.1 到 1，共 10 个点

    # 初始化延迟矩阵
    delay_matrix = np.zeros((len(p_i_values), len(eta_values)))

    # 模拟不同参数下的延迟
    for i, p_i in enumerate(p_i_values):
        for j, eta in enumerate(eta_values):
            delay_matrix[i, j] = env.simulate_delay(p_i, eta)

    # 绘制曲线图
    for i, p_i in enumerate(p_i_values):
        plt.plot(eta_values, delay_matrix[i, :], label=r'$P_i =$' +str (p_i)+ 'dBm')

    plt.xlabel('Local Training Accuracy η')
    plt.ylabel('Total Delay T')
    plt.title('Impact of Transmission Power and Local Training Accuracy on Total Delay')
    plt.legend()
    plt.grid(True)
    plt.show()


    # uav, task = env.reset()
    # T = env.System_Model()
    # print("无人机模型为：\n" + str(
    #     uav) + "\n注：其中的数据从左到右依次为无人机所处位置的x轴坐标、无人机所处位置的y轴坐标、移动方向、移动速度和本地计算能力\n")
    # print("任务模型为：\n" + str(task) + "\n注：其中的数据分别为不同传感器执行任务所采集的数据量大小\n")
    # print("执行数据采集处理任务的总时延为：" + str(T))

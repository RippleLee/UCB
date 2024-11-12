import torch
from torchvision import datasets, transforms


# 获取数据集
def get_dataset(dir, name):
    if name == 'mnist':
        # root(这里是dir): 数据路径
        # train参数表示是否是训练集(True)或者测试集(False)
        # download=true表示从互联网上下载数据集并把数据集放在root路径中
        # transform：图像类型的转换
        # 将图片转化为张量对象。通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在 0 到 1 之间
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())

    elif name == 'cifar':
        # transforms.Compose(图片预处理步骤)是将多个transform组合起来使用（由transform构成的列表）
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # 从图片中随机裁剪出尺寸为size的图片。size: 所需裁剪出的图片尺寸。padding: 设置填充大小，当为a时，上下左右均填充a个像素。
            transforms.RandomHorizontalFlip(),  # 水平翻转(左右翻转图像通常不会改变对象的类别。这是最早且最广泛使用的图像增广方法)
            transforms.ToTensor(),  # 将图片转化为张量对象。通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在 0 到 1 之间
            # transforms.Normalize: 标准化图像的每个通道。第一个参数: 均值; 第二个参数: 方差
            # 对于RGB(红、绿、蓝)颜色通道，我们分别标准化每个通道。具体而言，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                         transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset

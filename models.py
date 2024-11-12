import torch
from torchvision import models


def get_model(name, pretrained=True):
    # 卷积神经网络的训练是耗时的，很多场合不可能每次都从随机初始化参数开始训练网络。
    # pytorch中自带几种常用的深度学习网络预训练模型，如VGG、ResNet等。往往为了加快学习的进度，在训练的初期我们直接加载pre-train模型中预先训练好的参数
    if name == "resnet18":
        # 使用在ImageNet数据集上预训练的ResNet-18作为源模型。指定pretrained=True自动下载预训练的模型参数
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        # 使用在ImageNet数据集上预训练的ResNet-50作为源模型。指定pretrained=True自动下载预训练的模型参数
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model

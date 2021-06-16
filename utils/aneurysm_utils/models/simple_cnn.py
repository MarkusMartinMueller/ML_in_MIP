# -*- coding: utf-8 -*-
# 定义卷积神经网络
import torch.nn as nn
import torch.nn.functional as F
import torch

#  根据网络层的不同定义不同的初始化方式


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CNN3D(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        super(CNN3D, self).__init__()
        # in_channels, out_channels, kernel_size=3*3*3
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.BN3d1 = nn.BatchNorm3d(num_features=8)  # num_feature为输入数据通道数
        self.BN3d2 = nn.BatchNorm3d(num_features=16)
        self.BN3d3 = nn.BatchNorm3d(num_features=32)
        self.BN3d4 = nn.BatchNorm3d(num_features=64)
        self.BN3d5 = nn.BatchNorm3d(num_features=128)
        self.pool1 = nn.AdaptiveMaxPool3d((61, 73, 61))  # (61,73,61) is output size
        self.pool2 = nn.AdaptiveMaxPool3d((31, 37, 31))
        self.pool3 = nn.AdaptiveMaxPool3d((16, 19, 16))
        self.pool4 = nn.AdaptiveMaxPool3d((8, 10, 8))
        self.pool5 = nn.AdaptiveMaxPool3d((4, 5, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(10240, 1300)  # 接着三个全连接层
        self.fc2 = nn.Linear(1300, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.BN3d1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.BN3d2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.BN3d3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.BN3d4(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.BN3d5(self.conv5(x)))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)  # x.size(0)此处为batch size
        # x = x.view(-1, 173056)  #
        x = self.dropout(x)
        #if self.fc1 is None:
        #    self.fc1 = nn.Linear(x.shape(), 1300)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


#  根据网络层的不同定义不同的初始化方式


def weight_init_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv3d，使用相应的初始化方式
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SimpleCNN2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def __call__(self, x):
        x = self.conv1(x)
        print(x.size)
        return torch.squeeze(x)


class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.Sigmoid(),
        )

    def __call__(self, x):
        x = self.conv1(x)
        return x

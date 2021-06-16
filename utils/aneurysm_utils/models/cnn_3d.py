import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List

class CNN3DMoboehle(nn.Module):
    """Model used by moboehle with one dense layer less"""

    def __init__(self, input_shape: tuple, num_classes: int, dropout=0.4, dropout2=0.4):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.dense_1 = nn.Linear(calculate_fc_input_shape(
            64, 4, [2, 3, 2, 3], np.asarray(input_shape)), 64)
        self.dense_2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)
        return x


class ClassificationModel3D(nn.Module):
    # From JRieke
    """The model we use in the paper."""

    def __init__(self, input_shape: tuple, num_classes: int, dropout=0, dropout2=0):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        print(calculate_fc_input_shape(
            64, 4, [2, 3, 2, 3], np.asarray(input_shape)))
        self.dense_1 = nn.Linear(calculate_fc_input_shape(
            64, 4, [2, 3, 2, 3], np.asarray(input_shape)), 128)
        self.dense_2 = nn.Linear(128, 64)
        self.dense_3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = F.max_pool3d(x, 2)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = F.max_pool3d(x, 3)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = F.max_pool3d(x, 2)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = F.max_pool3d(x, 3)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.relu(self.dense_2(x))
        x = self.dense_3(x)

        # Note that no sigmoid is applied here, because the network is used in combination with BCEWithLogitsLoss,
        # which applies sigmoid and BCELoss at the same time to make it numerically stable.

        return x


class CNN3DSoftmax(nn.Module):
    """The same model as above but with one linear layer more"""

    def __init__(self, input_shape: tuple, num_classes: int, dropout=0.0, dropout2=0.0):
        nn.Module.__init__(self)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        # ToDo: nn.MaxPool3d(3)
        self.Conv_2_mp = nn.MaxPool3d(2)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(2)
        self.dense_1 = nn.Linear(
            calculate_fc_input_shape(64, 4, [2, 2, 2, 2], np.asarray(input_shape)), 128)
        self.dense_2 = nn.Linear(128, 64)
        self.dense_3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        x = self.dense_2(x)  # TODO: add relu? x = self.relu(self.dense_2(x))
        x = self.softmax(self.dense_3(x))
        return x


class CNN3DTutorial(nn.Module):
    """https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html"""

    def __init__(self,  input_shape: tuple, num_classes: int):
        nn.Module.__init__(self)
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(1, 6, 3)
        self.conv_1_mp = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(6, 16, 3)
        self.conv_2_mp = nn.MaxPool3d(2)
        # an affine operation: y = Wx + b
        # 6*6 from image dimensiongm
        self.fc1 = nn.Linear(calculate_fc_input_shape(
            16, 2, [2, 2], np.asarray(input_shape)), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv_1_mp(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = self.conv_2_mp(self.conv2(x))
        # ToDo: try different x.view x = x.view(-1, 16 * 4 * 4)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def calculate_fc_input_shape(num_channels: int, num_layers: int, kernel_sizes_pooling: List[int], input_shape: np.ndarray):
    """calculate output shape of image after last layer. If no pooling layer then 1 for the layer"""
    for i in range(num_layers):
        input_shape -= 2
        input_shape = input_shape // kernel_sizes_pooling[i]
    return num_channels * np.prod(input_shape)

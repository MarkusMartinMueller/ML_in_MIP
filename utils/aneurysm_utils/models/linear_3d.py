import torch.nn as nn

class LinearModel3D(nn.Module):
    """The same model as above but with one linear layer more"""

    def __init__(self, num_classes: int, input_shape: tuple):
        super(LinearModel3D, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

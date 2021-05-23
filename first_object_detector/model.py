import torch
from torch import nn

class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector,self).__init__()
        self.flatten =nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*8,200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200,4)
        )
    def forward(self,x):
        x =self.flatten(x)
        coordinates=self.linear_relu_stack(x)
        return coordinates
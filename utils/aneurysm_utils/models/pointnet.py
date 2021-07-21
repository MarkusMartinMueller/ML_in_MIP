import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
import nibabel as nib
from pathlib import Path
import numpy as np

"""
Implemenation of the pointnetplus plus plus paper https://arxiv.org/abs/1706.02413

Based on:
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py
"""

class SAModule(torch.nn.Module):
    """
    Implementation of SA Layer
    
    """
    def __init__(self, ratio, r, nn):
        """
        Parameters
        ----------
        ratio: how much of all points should be sampled
        r: radius in which points should be clustered
        nn: multilayer perceptron to use for the point convolution for feature upsampling
        """
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        """
        First does farthest point sampling with all given points. 
        Than group together all points within range to sampled points. See paper for more information
        
        Parameters
        ----------
        x: list of values
        pos: list of coordinates
        batch: batch indices showing to which batch a points belongs
        
        Returns
        -------
        processed inputs
        """
        
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global sample module
    """
    def __init__(self, nn):
        """
        Parameters
        ----------
        nn:Neural network used for featue upsampling
        """
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        """
        For each feature gets the maximum of all points and stores them im in x 
        
        Parameters
        ----------
        x: list of values
        pos: list of coordinates
        batch: batch indices showing to which batch a points belongs
        
        Returns
        -------
        processed inputs
        """
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    """
    Shared MLP for Feature upsampling or downsampling
    
    Parameters
    ----------
    Channels: array of integers to define number of features for each layer
    
    Returns
    -------
    MLP
    """
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ]
    )


class FPModule(torch.nn.Module):
    """
    Feature propagation layer
    
    """
    def __init__(self, k:int, nn):
        """
        Parameters
        ----------
        k: number of neighbours used for propagation
        nn:Neural network used for featue downsamplingsampling
        """
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        """
        Upsample number of points, features are interpolated of points out of layer before and feautures of the corresponding SA Layer 
        
        Parameters
        ----------
        x: list of values
        pos: list of coordinates
        batch: batch indices showing to which batch a points belongs
        x_skip: list of values of corresponding SA Layer 
        pos_skip: list of coordinates of corresponding SA layers, this coordinates are used to define the new points with the interpolated and added features
        batch_skip: batch indices showing to which batch a points belongs of the corresponding SA Layer
        """
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class SegNet(torch.nn.Module):
    """
    Implements all Layers defined before in architecture
    """
    def __init__(self, num_classes:int, dropout:float=0.5, start_radius:float=0.2, sample_rate1:float= 0.2, sample_rate2:float=0.25):
        """
        Paramters
        ---------
        num_classes: number of output classes
        dropout: dropout value for dropoout layers
        sample_rate1 and sample_rate2: sample rates for fps
        start_radius: first radius for grouping points 
        """
        super(SegNet, self).__init__()
        self.dropout = dropout
        self.sa1_module = SAModule(sample_rate1, start_radius, MLP([1 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(sample_rate2, start_radius * 2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 1, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        """
        Forward pass, also saves corresponding value in model for later evaluation
        
        Paramters
        ---------
        data: data in form of graph
        
        Returns
        -------
        torch vector with outputs
        """
        sa0_out = (data["x"], data["pos"], data["batch"])
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, pos, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        self.output_graph = Data(x=torch.squeeze(x),pos=pos)
        return torch.squeeze(x)


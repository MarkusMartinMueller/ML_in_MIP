#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

import numpy as np
from ipywidgets import widgets
import matplotlib.pyplot as plt
import nilearn.plotting as nip
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch.optim import Adam
from tqdm.notebook import tqdm, trange


# In[ ]:





# In[ ]:


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,3,1,1, bias = False), # batch, channels, depth,height,width
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels,out_channels,3,1,1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        
       
                
    def forward(self,x):
        return self.conv(x)

class DoubleConvUp(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConvUp,self).__init__()
        
        self.convUp = nn.Sequential(
            
            nn.Conv3d(in_channels,out_channels,3,1,1, bias = False), # batch, channels, depth,height,width
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,out_channels,3,1,1, bias = False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
        
       
                
    def forward(self,x):
        return self.convUp(x)

class Unet_3D(nn.Module):
    def __init__(self, in_channels= 1,out_channels= 2, filters = [64,128,256]):
        
        super(Unet_3D,self).__init__()
        self.up = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2)
    
        #Encoder of the U-net

        # In the paper, they convolve the image from 3 to 32 to 64
        self.conv32_64 = self.conv = nn.Sequential(
            nn.Conv3d(in_channels,32,3,1,1, bias = False), # batch, channels, depth,height,width
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32,64,3,1,1, bias = False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        
        in_channels = filters[0]
        for filter in filters:
          ### ignoring 64 because  i computed it in conv_32_64
            if filter ==filters[0]:
                continue
            self.down.append(DoubleConv(in_channels,filter)) ## add DoubleCon layer to the module list, maps 3 to 64, 64 to 128 ...
            in_channels = filter
        
        ## Decoder of the U-Net
    
        for filter in reversed(filters):
            self.up.append(nn.ConvTranspose3d(filter*2,filter, kernel_size = 2,stride=2)) ## features * 2 , adding the skip connection
            self.up.append(DoubleConvUp(filter*2,filter))
        
        
        ##Bottleneck / Bottom
    
        self.bottleneck = DoubleConv(filters[-1],filters[-1]*2)
    
        ## 1x1x1 Conv
    
        self.final_conv = nn.Conv3d(filters[0],out_channels,kernel_size= 1)
    
    
    
    def forward(self,x):
            skip_connections = []
            ## first convolutions
            x = self.conv32_64(x)
            skip_connections.append(x)
            x = self.pool(x)
            ## the 
            for down in self.down:
                
                x = down(x) ## double conv layer 
                skip_connections.append(x) # add resulat of the doubleconv layer to the list, order important
                x = self.pool(x)
            
            
           

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]# reverse the list, first element ,e.g 512 feature maps, bottom skip
            
            for idx in range(0,len(self.up),2):    #step size two, because of the transpose conv and DoubleConv operation
                
                x = self.up[idx](x)
                
                skip_connection = skip_connections[idx//2]  ## division by 2 because of the step size
                
                
                concat_skip = torch.cat((skip_connection,x),dim=1) # batch,channel, hight, width  , adding along channel dim
        
                x = self.up[idx+1](concat_skip)## upsampling step
                
            
            return self.final_conv(x)
        
        
def test():
    
    x = torch.rand((1,3,64,64,64)) ## batch,channel,height,width,depth
    
    model = Unet_3D(in_channels=1,out_channels=1)
    preds = model(x)
    
    print(preds.shape)
    print(image.shape)

    #assert preds.shape == image.shape
    
    


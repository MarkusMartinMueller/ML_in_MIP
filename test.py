import utils
import torch
import nibabel as nib
from pathlib import Path
import nilearn.plotting as nip
import numpy as np
from nibabel import Nifti1Image
from pointnet import SegNet
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data import DataLoader
from datasets import NiftiDataset

import torch.nn.functional as F
from utils import train
from utils import test





root_path = "/workspaces/project_med/data"
processed_dir="/workspaces/project_med/data"
dataset= NiftiDataset(root_path,processed_dir,forceprocessing=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SegNet(2).to(device)



# train_dataset = dataset[:int(len(dataset)*0.8)]

# test_dataset = dataset[int(len(dataset)*0.8):]
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
#                           num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
#                          num_workers=4)

# for epoch in range(1, 31):
#     train(model,train_loader,device)
#     accuracy = test(model,test_loader,device)
#     print('Epoch: {:02d}, accuracy: {:.4f}'.format(epoch, accuracy))


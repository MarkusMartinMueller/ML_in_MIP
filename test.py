import utils
import torch
import nibabel as nib
from pathlib import Path
import nilearn.plotting as nip
import numpy as np
from nibabel import Nifti1Image
from pointnet import SegNet
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from datasets import NiftiDataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from utils import train
from utils import test
from utils import graph_to_image




root_path = "/workspaces/project_med/data"
processed_dir="/workspaces/project_med/data/processed"

dataset= NiftiDataset(root_path,processed_dir,forceprocessing=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegNet(1).to(device)



train_dataset = dataset[:int(len(dataset)*0.8)]

test_dataset = dataset[int(len(dataset)*0.8):]
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                          num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=4)

# for epoch in range(1, 2):
#      train(model,train_loader,device)
    
model.load_state_dict(torch.load("modelweights.pt"))   
accuracy = test(model,test_loader,device)


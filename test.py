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

for epoch in range(1, 40):
    train(model,train_loader,device,pos_weight=50,lr=0.0001)
    accuracy = test(model,test_loader,device)
    print('Epoch: {:02d}, accuracy: {:.4f}'.format(epoch, accuracy))






# data_path = Path('data')
# image_nib = nib.load(data_path / 'A003_orig.nii.gz')
# image_aneurysm_nib = nib.load(data_path/'A003_masks.nii.gz')
# affine_aneurysm = image_aneurysm_nib.affine
# image = utils.min_max_normalize(image_nib.get_fdata())
# image_aneursysm = image_aneurysm_nib.get_fdata()
# affine_image = image_nib.affine
# mask =utils.intensity_segmentation(image,0.34)
# filtered= np.multiply(image,mask)
# filtered =utils.downsample(filtered,affine_image)
# image_aneursysm=utils.downsample(image_aneursysm,affine_aneurysm)
# graph = utils.segmented_image_to_graph(filtered,image_aneursysm,(47,47,40))
# m = torch.nn.Upsample(size=(256,256,220), mode='trilinear')
# image_new = utils.graph_to_image(graph,(47,47,40))
# print("test")
# nifti = Nifti1Image(image_new,affine=affine_image,header=image_nib.header)
# nip.view_img(image_new, bg_img=False, black_bg=True)

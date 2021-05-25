import utils
import torch
import nibabel as nib
from pathlib import Path
import nilearn.plotting as nip
import numpy as np
from nibabel import Nifti1Image
from pointnet import SegModel
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.data import DataLoader

data_path = Path("/workspaces/project_med/project/data")
image = utils.min_max_normalize(nib.load(data_path / 'A003_orig.nii').get_fdata())
image_aneursysm = nib.load(data_path / 'A003_masks.nii').get_fdata()
mask =utils.intensity_segmentation(image,0.34)
filtered= np.multiply(image,mask)
data = utils.segmented_image_to_graph(filtered,image_aneursysm)
data.batch=torch.zeros(data.y.shape,dtype = torch.int64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegNet(2).to(device)

y=model(data)
print(torch.exp(y))


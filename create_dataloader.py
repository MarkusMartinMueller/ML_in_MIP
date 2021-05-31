#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
import torch
import numpy as np
from ipywidgets import widgets
from monai.data import ArrayDataset,GridPatchDataset, DataLoader, PatchIter
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    ScaleIntensityd,
)
import numpy as np
import shutil
import os
import glob

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# In[ ]:


def prepare_dataset(data_dir):
    """
    data_path, string containing path to the training data
    
    train_images, list containing, the sorted paths to each of the training images
    train_labels, list containing, the sorted paths to each of the training masks
    
    data_dicts, list of dictionaris with keys image and label containing corresponding training pairs
    
    """
    
    
    
    
    
    train_images = sorted(
    glob.glob(os.path.join(data_dir,"*orig.nii.gz")))
    
    train_labels = sorted(
    glob.glob(os.path.join(data_dir, "*masks.nii.gz")))

    data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
    ]
    
    
    return train_images, train_labels, data_dicts
    





# In[ ]:


def create_dataloader(data_path,batch_size=1,num_workers=0,pin_memory=0):
    """
    data_path: path to the data storage, e.g. Path('../../../data/exploration-tutorial')
    
    
    return:
            dataloader with images and masks 
            images is a list with #batch_size tensor images (batch,channel,patch_height,patch_width,patch_depth), #batch_size tensor with shapes
            mask is a list with #batch_size tensor mask (batch,channel,patch_height,patch_width,patch_depth), #batch_size tensor with shapes
    
    """
    
    train_images, train_labels,data_dicts = prepare_dataset(data_path)
    
    transform = Compose([LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    Spacingd(keys=["image", "label"], pixdim=(
        1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])(data_dicts)
    
    
    train_img = []
    train_mask = []
    for i in range(len(transform)):
        train_img.append(transform[i]['image'])
        train_mask.append(transform[i]['label'])
        
    from monai.data import GridPatchDataset, DataLoader, PatchIter



    ds_np = ArrayDataset(train_img,seg = train_mask)
    patch_iter = PatchIter(patch_size=(28, 28, 28), start_pos=(0, 0, 0),mode='wrap')
    
    def img_seg_iter(x):
        return (zip(patch_iter(x[0]), patch_iter(x[1])),)

    ds = GridPatchDataset(ds_np, img_seg_iter, with_coordinates=False)




    
    
    
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory
    )

    return loader


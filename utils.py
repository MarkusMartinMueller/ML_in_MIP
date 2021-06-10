import numpy as np 
from nibabel import Nifti1Image
from skimage import color
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import median
from torch_geometric.data import Data, dataloader
import copy
import torch
from torch_geometric.utils import intersection_and_union as i_and_u
from pointnet import SegNet
import torch.nn.functional as F

from monai.transforms import Spacing
import torch.nn as nn

def intensity_segmentation(image : np.array,threshold :float) ->np.array:
    """
    Does a binary segmentation on an image depending on the intensity of the pixels, if the intentsity is bigger than the threshold its a vessel,
    else its background
    Parameters
    ----------
    image
        The image to be segmented
    threshold
        The intensity threshold 
    Returns
    -------
    numpy array
        A mask for the vessel
    
    """
    mask = copy.copy(image)
    mask[mask >threshold] = 1

    mask[mask<threshold] =0
    return mask
    
    
def min_max_normalize(mri_img):
    """Function which normalized the mri images with the min max method"""
    mri_img-= np.min(mri_img)
    mri_img /= np.max(mri_img)
    return mri_img
    
def segmented_image_to_graph(image : np.array, image_mask: np.array,shape:tuple=(256,256,220)):
    """
    Converts orinal image to labeled graph
    
    Parameters
    ----------
    image
        image with one channel for intensity values
    mask
        array of same size, with 1 for aneuyrism
    Returns
    -------
    torch_geometric.data.Data
        contains the graph
    """
    labels = torch.tensor(np.ndarray.flatten(image_mask[image!=0]),dtype = torch.float)
    coordinates =torch.tensor(normalize_matrix(np.where(image !=0),shape),dtype=torch.float)
    intensity_values =torch.unsqueeze( torch.tensor(np.ndarray.flatten(image[image!=0]),dtype=torch.float),1)
    #data_point_vector= np.vstack([intensity_values,coordinates])
    return Data(x=intensity_values,pos=coordinates,y=labels)


def normalize_matrix(x:np.array,shape:tuple):
    a=x[0]/shape[0]
    b=x[1]/shape[0]
    c=x[2]/shape[2]

    return np.column_stack((a,b,c))


def train(model,train_loader,device :torch.device,lr=0.001,pos_weight=6):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion =nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]).to(device))
    #criterion =nn.BCEWithLogitsLoss().to(device)
    #criterion =nn.CrossEntropyLoss(weight=torch.FloatTensor([100.0]).to(device))

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        if data == None: continue
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        

   
        loss = criterion(torch.unsqueeze(out,dim=1),torch.unsqueeze(data.y,dim=1))
        #loss = criterion(torch.squeeze(out),data.y)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
        with torch.no_grad():
            total_loss += float(loss.item())
            correct_nodes += int(torch.round(torch.sigmoid(out)).eq(data.y).sum().item())
            total_nodes += int(data.num_nodes)
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
            f'Train Acc: {correct_nodes / total_nodes:.4f}')
    torch.save(model.state_dict(), "modelweights.pt")

   
    

@torch.no_grad()
def test(model : SegNet,loader ,device :torch.device):
    model.eval()
    print("test")
    accuracies=[]
    
    for data in loader:
        data = data.to(device)
        pred = model(data)
        correct_nodes = (pred.x).eq(data.y).sum().item()
        total_nodes =int(data.num_nodes)
        accuracies.append(correct_nodes/total_nodes)
        torch.cuda.empty_cache
    return np.sum(accuracies)/len(accuracies)
        
def graph_to_image(graph : Data,shape:tuple=(256,256,220)):
    image =np.zeros(shape)
    graph.pos[:,:2]=graph.pos[:,:2]*shape[0]
    graph.pos[:,2:]=graph.pos[:,2:]*shape[2]
    for value, coords in zip(graph.x,graph.pos):
        coords=[round(num) for num in coords.tolist()]

        image[coords[0],coords[1],coords[2]]= value.item()
    return image


def downsample(struct_arr :np.array,affine :np.array,resample_dim:tuple=(1.5, 1.5, 1.5) ):
    struct_arr = struct_arr.astype("<f4")
    if resample_dim is not None:
        struct_arr = np.expand_dims(struct_arr, axis=0)
        spacing = Spacing(pixdim=resample_dim)
        struct_arr = spacing(struct_arr, affine)[0]
        struct_arr = np.squeeze(struct_arr, axis=0)
    return struct_arr

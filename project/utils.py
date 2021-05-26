import numpy as np 
from nibabel import Nifti1Image
from skimage import color
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import median
from torch_geometric.data import Data
import copy
import torch

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
    
def segmented_image_to_graph(image : np.array, image_mask: np.array):
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
    labels = torch.tensor(np.ndarray.flatten(image_mask[image!=0]),dtype = torch.long)
    coordinates =torch.tensor(normalize_matrix(np.where(image !=0)),dtype=torch.float).T
    intensity_values =torch.unsqueeze( torch.tensor(np.ndarray.flatten(image[image!=0]),dtype=torch.float),1)
    #data_point_vector= np.vstack([intensity_values,coordinates])
    return Data(x=intensity_values,pos=coordinates,y=labels)


def normalize_matrix(x:np.array):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    
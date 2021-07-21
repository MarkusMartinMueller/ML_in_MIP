import os
from typing import List,Tuple

import tqdm
import nilearn
import numpy as np
import scipy as sp
import copy
from nilearn.image import load_img, new_img_like, resample_to_img
from addict import Dict
import matplotlib.pyplot as plt
import nibabel as nib
import copy
from ast import literal_eval
import aneurysm_utils


def intensity_segmentation(mri_imgs: List[np.memmap], threshold: float) -> np.array:
    """
    Does a binary segmentation on an image depending on the intensity of the pixels, if the intentsity is bigger than the threshold its a vessel,
    else its background
    Parameters
    ----------
    image
        Images to be segmented
    threshold
        The intensity threshold
    Returns
    -------
    numpy array
        A mask for the vessel

    """
    mri_imgs = [np.where(image > threshold, image, 0) for image in mri_imgs]

    return mri_imgs



def resize_mri(img:np.array, size:Tuple, interpolation:int=0):
    """
    Resize img to size. 
    
    Parameters
    ----------
    size: desired size
    interpolation: Interpolation between 0 (no interpolation) and 5 (maximum interpolation).
    
    
    Returns
    ------
    The resized image
    """
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    # img_arr = img_arr.resize((256,256,128))
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)


def get_nift_like(env: aneurysm_utils.Environment, mri_imgs:List[np.array]):
    """
    Function tranforms list of numpy arrays into Niimg-like object"""
    template = get_mri_template(env, shape=mri_imgs[0].shape)

    if template is None:
        raise IndexError(
            "For this image size: "
            + str(mri_imgs[0].shape)
            + " no template is provided"
        )
    nifti_likes = []
    for img in mri_imgs:
        nifti_like = new_img_like(template, img)
        nifti_likes.append(nifti_like)
    return nifti_likes


def smooth_imgs(env: aneurysm_utils.Environment, mr_imgs: list, fwhm=1) -> list:
    """Smooth images."""

    niimg_likes = get_nift_like(env, mr_imgs)
    smoothed = []
    for img in niimg_likes:
        smoothed.append(
            nilearn.image.smooth_img(img, fwhm=fwhm).get_data().astype("<f4")
        )
    return smoothed


def get_reshaped_img(mri_imgs:List[np.array], new_shape:Tuple=(256, 256, 256), plot:bool=False):
    """
    Reshape the images into a new shape by default = (256, 256, 256, 1). Only input imgs with a smaller shape
    
    Paramters
    ---------
    mri_imgs: list of 3d images
    new_shape: reshape image to that shape
    plot: if true plots the new images
    
    Returns
    -------
    list of reshaped images
    
    """
    new_mri_imgs = []
    for img in mri_imgs:
        # fill image with zeros
        zeros = np.zeros(new_shape, dtype=np.int32)
        zeros[: img.shape[0], : img.shape[1], : img.shape[2]] = img[:, :, :, 0]
        new_mri_imgs.append(zeros)

        if plot:
            print_img(zeros)
    return new_mri_imgs



def print_img(mri_img:np.array, idx:Tuple=None):
    """
    Print single mri image at three axe
    
    Parameters
    ----------
    mri_imgs: image to print
    idx: tuple where the entries define where to slice the image in order z,y,x
    
    """
    if idx == None:
        idx = (
            int(mri_img.shape[0] / 2),
            int(mri_img.shape[1] / 2),
            int(mri_img.shape[2] / 2),
        )
    plt.subplot(1, 3, 1)
    plt.imshow(np.fliplr(mri_img[:, :, idx[2]]).T, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(np.flip(mri_img[:, idx[1], :]).T, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(np.fliplr(mri_img[idx[0], :, :]).T, cmap="gray")
    plt.show()



def min_max_normalize(mri_imgs: List[np.memmap]):
    """
    Function which normalize the mri images with the min max method
    
    Parameters
    ----------
    mri_imgs: list of images
    
    Returns
    -------
    list of normalized images
    """
    for i in range(len(mri_imgs)):
        mri_imgs[i] -= np.min(mri_imgs[i])
        mri_imgs[i] /= np.max(mri_imgs[i])
    return mri_imgs


def mean_std_normalize(mri_imgs: List[np.memmap]):
    """
    Function which normalize the mri images with  mean and standard dev
      
    Parameters
    ----------
    mri_imgs: list of images
    
    Returns
    -------
    list of normalized images
    """
    mean = np.mean(mri_imgs)
    std = np.std(mri_imgs)
    mri_imgs = (mri_imgs - mean) / std
    return mri_imgs


def check_mri_shapes(mri_imgs: list):
    """
    Checks all given shapes in image lists and prints them
    
    Parameters
    ----------
    mri_imgs: list of images
    
    Returns
    -------
    most common shape in list
    
    """
    from collections import Counter

    shape_cnt = Counter()
    for mri_img in mri_imgs:
        shape_cnt[str(mri_img.shape)] += 1
    print("Most common:")
    most_common_count=0
    for shape, count in shape_cnt.most_common():
        if count > most_common_count:
            most_common_count = count
            most_common_shape = shape

        print("%s: %7d" % (shape, count))


    return literal_eval(most_common_shape)


def is_int(val):
    """
    Check if val is int
    Parameters
    ----------
    val: value to check type
    
    Returns
    -------
    True or False
    """
    if type(val) == int:
        return True
    else:
        if val.is_integer():
            return True
        else:
            return False
        
def patch_list(data,patch_size):
    """
    Creates list of patch out of data
    
    Parameters
    ----------
    data: numpy.array
        containing dataset of dimensions (size_of_set,height,width,depth),e.g. (75,139,139,120)
    patch_size: int size of each patch
    
    Returns
    -------
    
    list_patch: list
        each element is one image of type numpy.array/torch.tensor with dimensions(n_classes,most_common_shape)
        most_common_shape: e.g. (139,139,120)
    """
    list_patch = []

    for n in range(len(data)):
        patch = patch_creater(data[n],patch_size)
        list_patch = list_patch+patch
    

    return list_patch

def patch_creater(image, patch_size):
    """
    Creates overlapping patches from  preprocessed image, the number of patches is fixed to certain value
    and the size can be changed as well.
    
    
    ----------
    image: numpy.array
        image which will be sliced into patches
    patch_size: tuple of int
        size of the patch, equal in each direction
   
    Returns
    -------
    numpy.array  (n_patches,channels,patch_size,patch_size,patch_size)
        list containing the patches

    """
    
    dim = np.array(image.shape)# size of the image
    
    # calculates the number of patches for each dim, to cover all voxel at least once
    #e.g for 139,139,120 with patch_size 64 the output would be (3,3,2)
    n_patches = np.ceil(dim/patch_size).astype(int) 
    
    
    # rest represents the entries which are overlapping with it's neighboring patch
    rest  = n_patches * patch_size%dim ## calculates the remaining voxel which are going to overlap 

    patches = []
    for i in range(n_patches[0]):
        
        if i == n_patches[0]-1: ## only the last cube is an overlapped cube
            start_x = i*patch_size-rest[0]## indices were to start and stop the slicing trough the image array
            stop_x= (i+1)* patch_size-rest[0]
              
        else:    
            start_x = i*patch_size
            stop_x = (i+1)* patch_size

        
              
        for j in range(n_patches[1]):
            if j == n_patches[1]-1: ## only the last cube is an overlapped cube
                start_y = j*patch_size-rest[1]
                stop_y= (j+1)* patch_size-rest[1]
              
            else:    
                start_y = j*patch_size
                stop_y = (j+1)* patch_size
            
            for k in range(n_patches[2]):
                if k == n_patches[2]-1: 
                    start_z = k*patch_size-rest[2]
                    stop_z = (k+1)* patch_size-rest[2]
              
                else:    
                    start_z = k*patch_size
                    stop_z = (k+1)* patch_size

                # appending patches starting from indices 0,0,0 to dim
                patches.append(image[start_x:stop_x,start_y:stop_y,start_z:stop_z])
        
    return patches


def preprocess(
    env: aneurysm_utils.Environment, mri_imgs: list, preprocessing_params: dict
) -> list:
    """
    Preprocess images with given parameters of dictionary
    
    Parameters
    ----------
    env: environment in which to do the preprocessing
    mri_imgs: list of 3d images
    preprocessing_params: parameters for preprocessing in this form( not all entries have to be there):
         {
             resample_to_mni152: boolean, if given resample to mni 152
             min_max_normalize: boolean, if true do min max normalize
             mean_std_normalize: boolean, if true do mean std normalize
             smooth_img: int or boolean, if true do smoothing if int do smoothing with this factor
             intensity_segmentation: float, if given do intensity segmentation with this threshold
             patch size:int, if given to deconstruct images in to patches with given size
        }
    """
    params = Dict(preprocessing_params)
    if params.resample_to_mni152:
        env.log.info("Preprocessing: Resample to MNI152...")
        mri_imgs = resample_to_mni152(env, mri_imgs)
    # TODO: change to numpy array .get_data().astype("<f4")
    if params.get_reshaped_img:
        env.log.info("Preprocessing: Reshape Image...")
        if "new_shape" not in params:
            params.new_shape = (256, 256, 256)
        mri_imgs = get_reshaped_img(mri_imgs, params.new_shape)
    if params.smooth_img:
        env.log.info("Preprocessing: Smooth Image...")
        if is_int(params.smooth_img):
            mri_imgs = smooth_imgs(env, mri_imgs, fwhm=int(params.smooth_img))
        else:
            mri_imgs = smooth_imgs(env, mri_imgs)
    if params.mean_std_normalize:
        env.log.info("Preprocessing: Mean STD Normalize...")
        mri_imgs = mean_std_normalize(mri_imgs)
    if params.min_max_normalize:
        env.log.info("Preprocessing: Min Max Normalize...")
        mri_imgs = min_max_normalize(mri_imgs)
    if params.get("intensity_segmentation"):
        env.log.info("Preprocessing: Intensity Segmentation...")
        mri_imgs = intensity_segmentation(mri_imgs, params.get("intensity_segmentation"))
    if params.get("patch_size"):
        patch_size = params.get("patch_size")
        env.log.info(f"Preprocessing: Creating patches in the size of {patch_size}...")
        mri_imgs = patch_list(mri_imgs, patch_size)
    return mri_imgs


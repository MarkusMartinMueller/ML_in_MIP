from sklearn.cluster import DBSCAN
from typing import List
import nibabel as nib
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import os
from monai.transforms import Resample
from scipy.ndimage import distance_transform_bf


def dbscan(mri_images:List[np.array]):
    new_mri_images=[]
    for image in mri_images:
        db = DBSCAN(eps=1, min_samples=3).fit(np.array(np.where(image==1)).T)
        labels =db.labels_
        empty= np.zeros(image.shape)
        for count,coords in enumerate(np.array(np.where(image==1)).T):
            empty[coords]=labels[count]+1
        new_mri_images.append(empty)
    return mri_images

def remove_border_candidates(mri_images:List[np.array]):
    for image in mri_images:
        borders= image.shape
        for cluster in range(1,np.unique(image)[-1]):
            coords=np.where(image==cluster)
            if np.equal(np.amax(coords,axis=0),borders) or np.min(coords)==0:
                image[image==cluster]=0


def resample(mri_images:List[np.array],dimension=(256,256,220)):
    resampler = Resample()
    new_mri_images=[]
    for image in mri_images:
        new_mri_images.append(resampler(image,np.zeros(dimension)))
    return new_mri_images

def bounding_boxes(mri_images:List[np.array]):
    bounding_boxes=[]
    for image in mri_images:
        boxes=[]
        for cluster in range(1,np.unique(image)[-1]):
            indices= np.where(image[image==cluster])
            boxes.append(np.max(indices,axis=0),np.min(indices,axis=0))
        bounding_boxes.append(boxes)
    return bounding_boxes


# def evaluate_dbscan(predicted:List[np.array],groundtruth:List[np.array]):
#     for image in predicted:
#         for cluster in range(1,np.unique(image)[-1]):
#             for


data_path = Path('../../datasets')
print(os.getcwd())
images=[]
for i in range(1,10):
    image_number= f"A00{i}"
    image_orig_path = image_number+'_orig.nii.gz'
    image_vessel_path=image_number+'_vessel.nii.gz'
    image_aneurysm_path =image_number+'_masks.nii.gz'
    try:
        images.append(nib.load(data_path/image_aneurysm_path).get_fdata())
    except:
        continue
for i in range(10,13):
    image_number= f"A0{i}"
    image_orig_path = image_number+'_orig.nii.gz'
    image_vessel_path=image_number+'_vessel.nii.gz'
    image_aneurysm_path =image_number+'_masks.nii.gz'
    try:
        images.append(nib.load(data_path/image_aneurysm_path).get_fdata())
    except:
        continue

dbscan(images)


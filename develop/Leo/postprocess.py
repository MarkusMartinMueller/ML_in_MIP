from sklearn.cluster import DBSCAN
from typing import List
import nibabel as nib
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import os


from sklearn.metrics.cluster import normalized_mutual_info_score
import open3d
from collections import defaultdict

from scipy.ndimage import zoom

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
"""
The MIT License (MIT)
Copyright (c) 2017 David Mugisha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    


"""
Computation of purity score with sklearn.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
import numpy as np
def draw_mask_3d(image:np.array):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(np.argwhere(image[0]).T[0],np.argwhere(image[0]).T[1],np.argwhere(image[0]).T[2],s=0.2,alpha=1)



def draw_bounding_box(vertice_points):
    Z=np.array(vertice_points)

    fig = plt.figure()
    ax = Axes3D(fig)
    verts= [(Z[0],Z[1]),(Z[0],Z[2]),(Z[0],Z[3]),(Z[6],Z[1]),(Z[7],Z[1]),(Z[2],Z[5]),
    (Z[2],Z[7]),(Z[3],Z[5]),(Z[3],Z[6]),(Z[4],Z[7]),(Z[4],Z[6]),(Z[4],Z[5])]

    for element in verts:
        x=[element[0][0],element[1][0]]
        y=[element[0][1],element[1][1]]
        z=[element[0][2],element[1][2]]
        ax.plot(x,y,z,c='r')


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
        for cluster in range(1,int(np.unique(image)[-1])+1):
            coords=np.where(image==cluster)
            if np.any(np.equal(np.amax(coords,axis=1)+1,borders)) or np.min(coords)==0:
                image[image==cluster]=0

    return mri_images


def resample(mri_images:List[np.array],dimension=(256,256,220)):
    new_mri_images=[]
    for image in mri_images:
        shape=image.shape
        zoom_factors= [dimension[0]/shape[0],dimension[1]/shape[1],dimension[2]/shape[2]]
        new_mri_images.append(zoom(image,zoom_factors))
    return new_mri_images

def bounding_boxes(mri_images:List[np.array]):
    bounding_boxes=[]
    for image in mri_images:
        boxes={"candidates":[]}
        for cluster in range(1,int(np.unique(image)[-1]+1)):

            box={}
            indices= open3d.utility.Vector3dVector(np.array(np.where(image==cluster)).T)
            oriented_box=open3d.geometry.OrientedBoundingBox.create_from_points(indices)

            box["position"]=oriented_box.get_center()                   
            box["extent"]= oriented_box.extent
            box["orthogonal_offset_vector"]=oriented_box.R
            box["box_object"]=oriented_box
            box["vertices"]=oriented_box.get_box_points()
            boxes["candidates"].append(box)

        bounding_boxes.append(boxes)
    return bounding_boxes


def evaluate_dbscan(predicted:List[np.array],groundtruth:List[np.array]):
    all_scores=[]
    for pred,truth in zip(predicted,groundtruth):
        clusterscores=defaultdict(list)
        for cluster in range(1,int(np.unique(pred)[-1])+1):
            indices =list(np.array(np.where(pred==cluster)).T)

            for true_cluster in range(1,int(np.unique(truth)[-1])+1):
                true_indices = list(np.array(np.where(truth==true_cluster)).T)

                clusterscores[str(cluster)].append(compare_two_list(true_indices,indices)/len(indices))
 
        all_scores.append(clusterscores)
    return all_scores

def compare_two_list(list_a,list_b):
    list_a= set([tuple(x)for x in list_a])
    list_b= set([tuple(x)for x in list_b])

    return len(list_a.intersection(list_b))

def min_max_normalize(mri_imgs: List[np.memmap]):
    """Function which normalized the mri images with the min max method"""
    for i in range(len(mri_imgs)):
        mri_imgs[i] -= np.min(mri_imgs[i])
        mri_imgs[i] /= np.max(mri_imgs[i])
    return mri_imgs

from skimage.filters import threshold_local

def local_intensity_segmentation(mri_imgs: List[np.array],block_size: int=35)->List[np.array]:
    for image in mri_imgs:

        for slice in range(0,image.shape[2]):
            local_thresh = threshold_local(image[:,:,slice], block_size, offset=0,method="gaussian")
            image[:,:,slice]=np.where(np.greater(image[:,:,slice],local_thresh),image[:,:,slice],0)
    return mri_imgs

def coverage(boxobjects:List,labeled_aneurysm_mask:np.array):
    total_score=0
    for label in range(1,np.unique(labeled_aneurysm_mask)+1):
        scorebefore=0
        aneurysm = np.where(labeled_aneurysm_mask==label)
        for box_dict in boxobjects:
            score= len(box_dict["box_object"].get_point_indices_within_boundinb_box(aneurysm))/len(aneurysm)
            if score>scorebefore:
                scorebefore=score
        total_score+=scorebefore
    
    return total_score/len(np.unique(labeled_aneurysm_mask)-1)

def bboxfit(boxobjects:List,labeled_aneurysm_mask:np.array,shape:tuple=(256,256,220)):
    total_score=0
    shape= labeled_aneurysm_mask.shape
    for label in range(1,np.unique(labeled_aneurysm_mask)+1):
        scorebefore=0
        aneurysm = np.where(labeled_aneurysm_mask==label)
        for box_dict in boxobjects:
            num_aneurysm_voxels_in_box=len(box_dict["box_object"].get_point_indices_within_boundinb_box(aneurysm))
            num_voxels_in_box = len( box_dict["box_object"].get_point_indices_within_boundinb_box(np.indices(shape)))
            score= num_aneurysm_voxels_in_box/num_voxels_in_box
            if score>scorebefore:
                scorebefore=score
        total_score+=scorebefore
    return total_score




# data_path = Path('../../datasets')
# print(os.getcwd())
# images=[]
# images_labeled_masks=[]
# for i in range(1,10):
#     image_number= f"A00{i}"
#     image_orig_path = image_number+'_orig.nii.gz'
#     image_vessel_path=image_number+'_vessel.nii.gz'
#     image_aneurysm_path =image_number+'_masks.nii.gz'
#     image_labeled_masks_path = image_number+'_labeledMasks.nii.gz'
#     try:
#         images.append(nib.load(data_path/image_aneurysm_path).get_fdata())
#         images_labeled_masks.append(nib.load(data_path/image_labeled_masks_path).get_fdata())
#     except:
#         continue
# for i in range(10,20):
#     image_number= f"A0{i}"
#     image_orig_path = image_number+'_orig.nii.gz'
#     image_vessel_path=image_number+'_vessel.nii.gz'
#     image_aneurysm_path =image_number+'_masks.nii.gz'
#     image_labeled_masks_path = image_number+'_labeledMasks.nii.gz'
#     try:
#         images.append(nib.load(data_path/image_aneurysm_path).get_fdata())
#         images_labeled_masks.append(nib.load(data_path/image_labeled_masks_path).get_fdata())
#     except:
#         continue


# images_masks =dbscan(images)
# boxes =bounding_boxes(images_masks)

# draw_bounding_box(boxes[0]["candidates"][0]["vertices"])

import os
from pathlib import Path
from typing import List
import numpy as np
from ipywidgets import widgets
import matplotlib.pyplot as plt
import nilearn.plotting as nip
import nibabel as nib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


data_path = Path('../../datasets')
vieworder=(2,1,0)
for i in range(10,20):
    image_number= f"A0{i}"
    image_orig_path = image_number+'_orig.nii.gz'
    image_vessel_path=image_number+'_vessel.nii.gz'
    image_aneurysm_path =image_number+'_masks.nii.gz'



    try:
        vessel_mask = nib.load(data_path/image_vessel_path)
    except:
        continue
    orig_image = nib.load(data_path/image_orig_path)
    orig_image_data=min_max_normalize([orig_image.get_fdata()])
    data = min_max_normalize([vessel_mask.get_fdata()]) 
    aneurysm_image = nib.load(data_path/image_aneurysm_path)
    aneurysm_data= min_max_normalize([aneurysm_image.get_fdata()])
    #segmented = intensity_segmentation(orig_image_data,0.1)
    segmented = local_intensity_segmentation(orig_image_data)                                 
    graph_data = segmented_image_to_graph(segmented[0],aneurysm_data[0])




#TODO 
#scores implementieren??
#notebook integrieren

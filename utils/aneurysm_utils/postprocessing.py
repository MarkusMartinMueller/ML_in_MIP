from scipy.ndimage import zoom
from sklearn.cluster import DBSCAN
from typing import List
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List
import aneurysm_utils
from addict import Dict
from aneurysm_utils.evaluation import evaluate_dbscan
import open3d
import json
from pathlib import Path
import nibabel as nib
import os
from nibabel.affines import apply_affine

def dbscan(mri_images:List[np.array],eps=3,min_samples=30):
    new_mri_images=[]
    for image in mri_images:
        indices= np.array(np.where(image==1)).T
        if len(indices)==0:
            print("No aneurysms found")
            new_mri_images.append(image)
            continue
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(indices))

        labels =db.labels_
        empty= np.zeros(image.shape)
        for count,coords in enumerate(indices):
            empty[coords[0],coords[1],coords[2]]=labels[count]+1
        new_mri_images.append(empty)
    del mri_images
    return new_mri_images

def remove_border_candidates(mri_images:List[np.array],offset):
    new_mri_images=[]
    for image in mri_images:
        borders= image.shape
        borders= (borders[0]-offset,borders[1]-offset,borders[2]-offset)

        for cluster in np.unique(image):
            if cluster==0:
                continue
            coords=np.where(image==cluster)
            if np.any(np.equal(np.amax(coords,axis=1)+1,borders)) or np.min(coords)==offset:
                image[image==cluster]=0
        for count,element in enumerate(np.unique(image)):
            image[image==element]=count
        new_mri_images.append(image)
    del mri_images
    return new_mri_images

def remove_noise(mri_images:List[np.array],size_smallest,size_biggest):
    new_mri_images=[]
    for image in mri_images:
        for cluster in range(1,int(np.unique(image)[-1])+1):
            volumen= np.array(np.where(image==cluster)).T.shape[0]
            if volumen<size_smallest:
                image=np.where(image==cluster,0,image)
            if volumen>size_biggest:
                image=np.where(image==cluster,0,image)
        new_mri_images.append(image)
    del mri_images
    return new_mri_images

def resample(mri_images:List[np.array],dimension=(256,256,220),order:int=0,binary:bool=False):
    new_mri_images=[]
    for image in mri_images:
        shape=image.shape
        zoom_factors= [dimension[0]/shape[0],dimension[1]/shape[1],dimension[2]/shape[2]]
        new_image= zoom(image,zoom_factors,order=order)
        if binary:
            new_image=np.rint(new_image)
        new_mri_images.append(new_image)
    del mri_images
    return new_mri_images

def bounding_boxes(mri_images:List[np.array],cases:List[str]=None):
    bounding_boxes=[]
    for count,image in enumerate(mri_images):
        boxes={"candidates":[]}
        if cases != None:
            boxes["dataset_id"]=cases[count]
        for cluster in range(1,int(np.unique(image)[-1]+1)):

            box={}
            indices= open3d.utility.Vector3dVector(np.array(np.where(image==cluster)).T)
            oriented_box=open3d.geometry.OrientedBoundingBox.create_from_points(indices)

            box["position"]=oriented_box.get_center().tolist()                   
            box["object_oriented_bounding_box"]={
                "extent":oriented_box.extent.tolist(),
                "orthogonal_offset_vectors":oriented_box.R.tolist()
                }


            box["vertices"]=oriented_box.get_box_points()
            boxes["candidates"].append(box)

        bounding_boxes.append(boxes)
    return bounding_boxes



def create_nifits(mri_images,cases,path_cada="../../cada-challenge-master/cada_segmentation/test-gt/",path_datasets='../../datasets/'):
    data_path = Path(path_datasets)
    for count,image in enumerate(mri_images):
        affine = nib.load(data_path / f'{cases[count]}_orig.nii.gz').affine
        img = nib.Nifti1Image(image, affine)
        img.to_filename(os.path.join(path_cada,f'{cases[count]}_labeledMasks.nii.gz'))

def create_task_one_json(images,cases,processing_times=None,path="reference.json",path_datasets='../../datasets/'):
    data_path = Path(path_datasets)
    dicto={
    "grand_challenge_username": "cake",
    "used_hardware_specification": {
        "CPU": "Intel Core i9 9900K 8x 3.60GHz",
        "GPU": "NVIDIA RTX 2080 Ti",
        "#GPUs": 1,
        "RAM_in_GB": 10,
        "additional_remarks": "special hardware requirements, other comments"
    }
    }
    dicto["task_1_results"]=[]
    for count,image in enumerate( images):
        box_entry={}
        affine =nib.load(data_path / f'{cases[count]}_orig.nii.gz').affine
        box_entry["dataset_id"]= cases[count]
        if processing_times !=None:
            box_entry["processing_time_in_seconds"]=processing_times[count]
        else:
            box_entry["processing_time_in_seconds"]=22.7
        box_entry["candidates"]=[]
        for cluster in range(1,int(np.unique(image)[-1]+1)):   
            box_entry["candidates"].append(create_score_bounding_box(image,affine,cluster))

        dicto["task_1_results"].append(box_entry)
    with open(path,"w") as f:
        json.dump(dicto,f)

def create_score_bounding_box(image,affine,cluster):
    indices = np.array(np.where(image == cluster)).T
    indices_ext = np.concatenate((indices, np.ones((indices.shape[0], 1))), axis=1)
    world_indices = (affine @ indices_ext.T).T[:, :3]
    indices_vector= open3d.utility.Vector3dVector(np.array(world_indices))
    oriented_box=open3d.geometry.OrientedBoundingBox.create_from_points(indices_vector)
    box_dict={}
    box_dict["position"]=oriented_box.get_center().tolist()                   
    box_dict["object_oriented_bounding_box"]={
                "extent":oriented_box.extent.tolist(),
                "orthogonal_offset_vectors":(-oriented_box.R).tolist()
                }
    return box_dict



def postprocess(
    env: aneurysm_utils.Environment, mri_imgs: list, preprocessing_params: dict
) -> list:
    params = Dict(preprocessing_params)
    if params.dbscan:
        env.log.info("Postprocessing: DBSCAN...")
        if "eps" not in params:
            params["eps"]=3
        if "min_samples" not in params:
            params["min_samples"]=30
            
        mri_imgs = dbscan(mri_imgs,params["eps"],params["min_samples"])
        if "size" not in params:
            params["size"]= 90
        env.log.info("Postprocessing: Removing noise...")
        mri_imgs = remove_noise(mri_imgs,params["size_smallest"],params["size_biggest"])
        
    if params.evaluate_dbscan:
        if "invidual_aneurysm_labels" not in params or "cases" not in params:
            env.log.info("Postprocessing: Can not evaluate DBSCAN because no ground truth or case list were given")
        else:
            env.log.info("Postprocessing: Evaluating DBSCAN")
            env.log.info(evaluate_dbscan(mri_imgs,params.invidual_aneurysm_labels,params.cases))
            
    if params.remove_border_candidates:
        if "offset" not in params:
            params.offset=0
        env.log.info("Postprocessing: Removing border candidates...")
        mri_imgs=remove_border_candidates(mri_imgs,params["offset"])  
    if params.resample:
        if "order" not in params:
            params.order=0
        if "resample_size" not in params:
            params.resample_size=(256,256,220)
        env.log.info(f"Postprocessing: Resample to Size{params.resample_size}")
        mri_imgs = resample(mri_imgs,dimension=params.resample_size,order=params.order,binary=True)
    return mri_imgs

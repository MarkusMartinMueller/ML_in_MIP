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

def dbscan(mri_images:List[np.array]):
    new_mri_images=[]
    for image in mri_images:
        db = DBSCAN(eps=2, min_samples=30).fit(np.array(np.where(image==1)).T)
        labels =db.labels_
        empty= np.zeros(image.shape)
        print(f"labels={np.unique(labels)}")
        for count,coords in enumerate(np.array(np.where(image==1)).T):
            print(coords)
            print(labels[count]+1)
            empty[coords]=labels[count]+1
        new_mri_images.append(empty)
    return new_mri_images

def remove_border_candidates(mri_images:List[np.array],offset):
    for image in mri_images:
        borders= image.shape
        borders= (borders[0]-offset,borders[1]-offset,borders[2]-offset)
        print(np.unique(image))
        for cluster in range(1,int(np.unique(image)[-1])+1):
            print(cluster)
            coords=np.array(np.where(image==cluster)).T
            print(coords)
            print(f"coords shape:{coords.shape}")
            if np.any(np.equal(np.amax(coords,axis=1)+1,borders)) or np.min(coords)==offset:
                image[image==cluster]=0

    return mri_images

def remove_noise(mri_images:List[np.array],size):
    for image in mri_images:
        print(np.unique(image))
        for cluster in range(1,int(np.unique(image)[-1])+1):
            volumen= len(np.array(np.where(image==cluster)).T)
            print(volumen)
            print(f"volumen{volumen}")
            if volumen<size:
                image=np.where(image==cluster,0,image)
        for element in range(0,size(np.unique(image))):
            image[image==element]=element
    return mri_images

def resample(mri_images:List[np.array],dimension=(256,256,220),order:int=3,binary:bool=False):
    new_mri_images=[]
    for image in mri_images:
        shape=image.shape
        zoom_factors= [dimension[0]/shape[0],dimension[1]/shape[1],dimension[2]/shape[2]]
        new_image= zoom(image,zoom_factors,order=order)
        if binary:
            new_image=np.rint(new_image)
        new_mri_images.append(new_image)
    return new_mri_images

def bounding_boxes(mri_images:List[np.array],cases:List[str]=None):
    bounding_boxes=[]
    for count,image in enumerate(mri_images):
        boxes={"candidates":[]}
        if cases != None:
            boxes["dataset_id"]=cases[count]
        for cluster in range(1,int(np.unique(image)[-1]+1)):
            print(np.unique(image))
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

def postprocess(
    env: aneurysm_utils.Environment, mri_imgs: list, preprocessing_params: dict
) -> list:
    params = Dict(preprocessing_params)
    if params.dbscan:
        env.log.info("Postprocessing: DBSCAN...")
        mri_imgs = dbscan(mri_imgs)
        if "size" not in params:
            params["size"]= 90
        mri_imgs = remove_noise(mri_imgs,params["size"])
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
            params.order=2
        if "resample_size" not in params:
            params.resample_size=(256,256,220)
        env.log.info(f"Postprocessing: Resample to Size{params.resample_size}")
        mri_imgs = resample(mri_imgs,dimension=params.resample_size,order=params.order,binary=True)
    return mri_imgs
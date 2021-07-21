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


def dbscan(mri_images:List[np.array],eps:int=3,min_samples:int=30):
    """
    Performs dbscan on all images of list with given parameters
    
    Parameters
    ----------
    mri_images:List of 3d-images in binary format, only zeros and ones
    eps: range in which dbscan should search for neighbors
    min_samples: minimum number of neighbors that a point should have for being a core point
    
    Returns
    -------
    List[np.array] : list of images with labels, 0 is background
    """
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

def remove_border_candidates(mri_images:List[np.array],offset:int):
    """
    Removes border candidates on all images of list with given parameters, also relabels images to have 
    sequencial labels e.g 0,1,2..
    
    Parameters
    ----------
    mri_images:List of 3d-images voxel should only have integers as values
    offset: how far away from the border the images should be removed

    
    Returns
    -------
    List[np.array] : list of 3d images 
    """
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

def remove_noise(mri_images:List[np.array],size_smallest:int,size_biggest:int):
    """
    Remove all clusters on 3d images that has less voxels or more voxels than desired
    
    Parameters
    ----------
    mri_images:List of 3d-images
    size_smallest: lower boundary of cluster size
    size_biggest: upper boundary of cluster size

    Returns
    -------
    List[np.array] : list of 3d images 
    """
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

def resample(mri_images:List[np.array],dimensions=(256,256,220),order:int=0,binary:bool=False):
    """
    Resamples images in list to given dimensions
    
    Parameters
    ----------
    mri_images:List of 3d-images
    dimensions: can be either a tuple if all images should be resized to the same dimension or list with entries for each image
    order: order of interpolation 0= nearest neighbor, 1= bilinear,2=cubic
    binary: if true rounds values 

    Returns
    -------
    List[np.array] : list of 3d images 
    """
    new_mri_images=[]
    
    for count,image in enumerate(mri_images):
        if isinstance(dimensions,list):
            dimension=dimensions[count]
        else:
            dimension =dimensions
        shape=image.shape
        zoom_factors= [dimension[0]/shape[0],dimension[1]/shape[1],dimension[2]/shape[2]]
        new_image= zoom(image,zoom_factors,order=order)
        if binary:
            new_image=np.rint(new_image)
        new_mri_images.append(new_image)
    del mri_images
    return new_mri_images

def bounding_boxes(mri_images:List[np.array],cases:List[str]=None):
    """
    Creates bounding boxes for visualization purposes for each image and saves a list of dictionaries with bounding box data.
    Saved dictionary has format {"candidates:[
        {
            position: Tuple,
            object_oriented_bounding_box:
                {
                    extent:Tuple,
                    orthogonal_offset_vectors:List[np.array]}}]"
                }
            vertices:open3d.utility.Vector3dVector
        }.. 
        ]
    with one list entry per candidate in image
    Parameters
    ----------
    mri_images:List of 3d-images voxel should only have integers as values
    cases: if given each dictionary in list has additional entry with dataset_id corresponding to entry in cases
    
    Returns
    -------
    List of dicitonaries as defined above
    """
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



def create_nifits(mri_images:List[np.array],cases:List[str],path_cada:str="../../cada-challenge-master/cada_segmentation/test-gt/",path_datasets:str='../../datasets/'):
    """
    Creates nifis out of a list of images
    
    Parameters
    ----------
    mri_images: List of 3d images
    cases: list of casenumbers which are corresponding to the images, for example ["A010", "A202",..]
    path_cada: where to save the niftis
    path_datasets: where to get the affines of the original niftis from
    """
    data_path = Path(path_datasets)
    for count,image in enumerate(mri_images):
        affine = nib.load(data_path / f'{cases[count]}_orig.nii.gz').affine
        img = nib.Nifti1Image(image, affine)
        img.to_filename(os.path.join(path_cada,f'{cases[count]}_labeledMasks.nii.gz'))

def create_task_one_json(images:List[np.array],cases:List[str],processing_times:float=None,path:str="reference.json",path_datasets:str='../../datasets/'):
    """
    Creates the json for the first task in the desired cada challenge format https://cada.grand-challenge.org/Submission-Details/
    Parameters:
    images: list of 3d images, should only have integer values order from 0 to number of clusters for example 0,1,2
    cases: list of casenumbers which are corresponding to the images, for example ["A010", "A202",..]
    processing_times: if given saves for each image entry the processing time 
    path: where to save the json
    path_datasets: where to get the affines of the original niftis from
    """
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

def create_score_bounding_box(image:np.array,affine:np.array,cluster:int):
    """
    Creates the bounding box in world coordinates of the given image and stores its parameters in json in cada challenge format 
    https://cada.grand-challenge.org/Submission-Details/
    
    Parameters
    ----------
    image: 3 dimensional np.array
    affine: affine of the original nifti image to transform image in woorld coordinates
    cluster: use entries with this value for creating the bounding box
    
    Returns
    -------
    Json file
    """
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

#dictionary which the voxel sizes to the corresponding resolution
dimensions_mapping={
    (1.0,1.0,1.0) : (139,139,120),
    (1.2,1.2,1.2) :(116,116,100),
    None: (256,256,220)
}


def unify_alternativ(image, patch_size,most_common_shape):
    """
    
    Gets as input an numpy array with shape number_of_patches,height,width,depth. It unifies
    the patches to create on numpy array of shape: most_common_shape. The overlapping cases are resolved
    by taking the maximun between the two overlapping values. In the non-overlapping case it's taking the maximum
    of zero and the value in the patch array.
    
    Parameters
    ----------
    image: numpy.array
          shape is (number_of_patches,h,w,d) 
    patch_size: int
            e.g. 64
    most_common_shape: tuple
                for example(139,139,120)

    Return

    image: np.array

          unified and unpatch image, shape (most_common_shape)


    """
    number_of_patches,heigth,width,depth = image.shape

    dim = np.array(most_common_shape)# size of the image
    n_patches = np.ceil(dim/patch_size).astype(int) # calculates the number of patches for each dim, to cover all voxel at least once in form e.g[3,3,2]
    rest  = n_patches * patch_size%dim ## calculates number of entries for each dimension which overlapp, means for example n_patches = 18 and 64 we have rest = 53
    
    h,w,d = most_common_shape

    ## initializing empty array for the unified image
    unified_image =  np.zeros([h,w,d])

    counter = 0 ## counter for patches
    

    
    for i in range(n_patches[0]):

        if i == n_patches[0]-1: ## only the last cube is an overlapped cube
          start_x = i*patch_size-rest[0]
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


              

              ###includes overlapping case
              #unified_image[start_x:stop_x,start_y:stop_y,start_z:stop_z] = np.maximum(unified_image[start_x:stop_x,start_y:stop_y,start_z:stop_z],max_class)
              ## take max of the patch and the zero array , which are at that same place in the reshape image
                unified_image[start_x:stop_x,start_y:stop_y,start_z:stop_z] = np.maximum(unified_image[start_x:stop_x,start_y:stop_y,start_z:stop_z],image[counter,:,:,:])
                
                counter+=1## next patch
                
                if (counter== number_of_patches):
                    break;
    
    return unified_image






def patch_unifier_alternativ(list_patches,size_test_set,most_common_shape,patch_size):
  """
  
  
  Creates a list of unpatches/unified images. Each image(numpy array) is unified with it's corresponding patches,which
  results in an numpy array with same shape as most_common_shape
  
  Parameters
  -------------------------
  list_patches: list
                containing predictions from the evaluations, length should be number_of_patches x length_test_set
                each element should have the form (h,w,d)
  size_test_set: int

  most_common_shape: tuple
                most_common shape from the original input images before patchifying
  
  patch_size: int



  Return:

  unified_images: list
                contains the unpatched images 
                each element has the shape:(most_common_shape) 

  """
  
  dim = np.array(most_common_shape)
  n_patches = np.ceil(dim/patch_size).astype(int)# output is number of patches per dimension

  number_of_patches = np.prod(n_patches)# number of patches overall

  h,w,d = most_common_shape
  
  unified_images = []
  assert (len(list_patches)/(number_of_patches))== size_test_set
  

  ##output list: each element has the form (number_of_patches,n_classes,h,w,d)
  images = np.split(np.array(list_patches),size_test_set)  

  for n in range(size_test_set):
        
        unified_images.append(unify_alternativ(images[n],patch_size,most_common_shape))
  assert len(unified_images) == size_test_set
  return unified_images


def postprocess(env: aneurysm_utils.Environment, mri_imgs: List[np.array], postprocessing_params: dict) -> list:
    """
    Postprocess list of images with given parameters of postprocessing dictionary
    
    Parameters
    ----------
    env: enviroment in which the postprocessing is done 
    mri_imgs: list of images 
    postprocessing_params: dictionary in form: (not all entries are neccessary)
        {
            patch_size: int, if given reconstruct images out of patches,
            num_imgs:int, number of images before depatching needed when patch_size is given
            resample_voxel_dim: float, uses corresponding tuple of dimensions_mapping for patch reconstruction
            dbscan: boolean,if true performs a dbscan and removes noise functions
            eps: int, dbscan parameter
            min_samples: int, dbscan_parameter
            size_smallest: int, remove noise parameter
            size_biggest: int, remove noise parameter
            remove_border_candidates: boolean, if true performs this function
            offset: int, paramter for remove_border_candidates
            resample:boolean, if true resamples the image
            order: int,paramter for resample
            resample_size: tuple or list of tuples, paramter for resample
        }
    
    """
    
    params = Dict(postprocessing_params)
    if params.patch_size:
        env.log.info("Postprocessing: Combine Patches...")
        mri_imgs = patch_unifier_alternativ(mri_imgs,params["num_imgs"],dimensions_mapping[params["resample_voxel_dim"]],params["patch_size"])
    if params.dbscan:
        env.log.info("Postprocessing: DBSCAN...")
        if "eps" not in params:
            params["eps"]=3
        if "min_samples" not in params:
            params["min_samples"]=30
            
        mri_imgs = dbscan(mri_imgs,params["eps"],params["min_samples"])
        env.log.info("Postprocessing: Removing noise...")
        mri_imgs = remove_noise(mri_imgs,params["size_smallest"],params["size_biggest"])
                  
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
        mri_imgs = resample(mri_imgs,dimensions=params.resample_size,order=params.order,binary=True)
    return mri_imgs

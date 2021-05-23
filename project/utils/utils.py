import numpy as np 
from nibabel import Nifti1Image
from skimage import color
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import median

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
    mask = min_max_normalize([image])[0]

    mask[mask >threshold] = 1

    mask[mask<threshold] =0
    return mask
    
    
def min_max_normalize(mri_imgs):
    """Function which normalized the mri images with the min max method"""
    for i in range(len(mri_imgs)):
        mri_imgs[i] -= np.min(mri_imgs[i])
        mri_imgs[i] /= np.max(mri_imgs[i])
    return mri_imgs
    
    

    
    
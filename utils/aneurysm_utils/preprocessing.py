import os
from typing import List

import tqdm
import nilearn
import numpy as np
import scipy as sp
import copy
from nilearn.image import load_img, new_img_like, resample_to_img
from addict import Dict
import matplotlib.pyplot as plt
import nibabel as nib

import aneurysm_utils


def intensity_segmentation(mri_imgs: List[np.memmap], threshold: float) -> np.array:
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
    segmented = []
    for image in mri_imgs:
        mask = copy.copy(image)
        mask[mask > threshold] = 1
        mask[mask < threshold] = 0
        segmented.append(mask)
    return segmented


def resize_mri(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    # img_arr = img_arr.resize((256,256,128))
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)


def get_mri_template(env: aneurysm_utils.Environment, shape):
    # TODO: check shape and modality?

    cache_key = "mri_template_img"
    if cache_key in env.cached_data:
        return env.cached_data[cache_key]
    else:
        if "mri_template_path" in env.cached_data:
            if not os.path.exists(env.cached_data["mri_template_path"]):
                print("Template path does not exist.")
                return None
            # Lazy load
            env.cached_data[cache_key] = load_img(
                env.cached_data["mri_template_path"], wildcards=True, dtype=None
            )
            return env.cached_data[cache_key]
        else:
            print("No valid mri template was found.")
            return None


def get_mri_mask(env: aneurysm_utils.Environment):
    # TODO: Adjust and understand
    cache_key = "mri_mask"
    if cache_key in env.cached_data:
        return env.cached_data[cache_key]
    else:
        # Lazy load the template
        mri_masks_path = aneurysm_utils.data_collection.get_mri_masks_file(env)
        mri_mask_path = os.path.join(mri_masks_path, "mask_T1.nii")
        return None

        env.cached_data[cache_key] = load_img(mri_mask_path).get_fdata()
        return env.cached_data[cache_key]


def get_nift_like(env: aneurysm_utils.Environment, mri_imgs):
    """Function tranforms list of numpy arrays into Niimg-like object"""
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


def get_reshaped_img(mri_imgs, new_shape=(256, 256, 256), plot=False):
    """Reshape the images into a new shape by default = (256, 256, 256, 1). Only input imgs with a smaller shape"""
    new_mri_imgs = []
    for img in mri_imgs:
        # fill image with zeros
        zeros = np.zeros(new_shape, dtype=np.int32)
        zeros[: img.shape[0], : img.shape[1], : img.shape[2]] = img[:, :, :, 0]
        new_mri_imgs.append(zeros)

        if plot:
            print_img(zeros)
    return new_mri_imgs


def get_roi_volume(env, mri_imgs, atlas=None):
    """Calculates the Volume of each ROI using the pauli 2017 template"""
    if atlas is None:
        atlas = "pauli_2017"

    if type(atlas) == str:
        atlas = [atlas]

    assert all(item in ["pauli_2017", "HOCPA_th25"] for item in atlas)

    if "pauli_2017" in atlas:
        pauli_atlas = nilearn.datasets.fetch_atlas_pauli_2017()
        s_struct_arr = nib.load(pauli_atlas["maps"]).get_data()
        s_labels = len(pauli_atlas["labels"])
    if "HOCPA_th25" in atlas:
        c_struct_arr = nib.load(
            env.get_file(
                "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_atlas-HOCPA_desc-th25_dseg.nii.gz"
            )
        ).get_data()
        c_labels = np.max(c_struct_arr)

    feature_list = []
    for mri_img in tqdm.tqdm(mri_imgs):
        features = []

        if "pauli_2017" in atlas:
            for label in range(s_labels):
                region = s_struct_arr[:, :, :, label] * mri_img
                volume = np.mean(region[np.nonzero(region)])
                features.append(volume)
        if "HOCPA_th25" in atlas:
            for label in range(c_labels):
                volume = np.mean(mri_img[np.where(c_struct_arr == (label + 1))])
                features.append(volume)
        feature_list.append(features)
    return feature_list


def print_img(mri_img, idx=None):
    """Print single mri image at three axes"""
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


def resample_to_mni152(env: aneurysm_utils.Environment, nii_imgs: list) -> list:
    """Function to resample 3D nii images in a list to (91, 109, 91)"""

    template = nilearn.datasets.load_mni152_template()
    resampled = []
    for img in nii_imgs:
        img_resamp = resample_to_img(img, template)
        resampled.append(img_resamp.get_fdata())
    return resampled


def min_max_normalize(mri_imgs: List[np.memmap]):
    """Function which normalized the mri images with the min max method"""
    for i in range(len(mri_imgs)):
        mri_imgs[i] -= np.min(mri_imgs[i])
        mri_imgs[i] /= np.max(mri_imgs[i])
    return mri_imgs


def mean_std_normalize(mri_imgs: List[np.memmap]):
    """Function which normalized the mri images with the min max method"""
    mean = np.mean(mri_imgs)
    std = np.std(mri_imgs)
    mri_imgs = (mri_imgs - mean) / std
    return mri_imgs


def check_mri_shapes(mri_imgs: list):
    from collections import Counter

    shape_cnt = Counter()
    for mri_img in mri_imgs:
        shape_cnt[str(mri_img.shape)] += 1
    print("Most common:")
    for shape, count in shape_cnt.most_common():
        print("%s: %7d" % (shape, count))

    """
    for i, mri_img in enumerate(mri_imgs):
        if not shape:
            shape = mri_img.shape
            print("First MRI shape: " + str(mri_img.shape))
        if mri_img.shape != shape:
            print(
                "Shape of MRI image "
                + str(i)
                + " has different shape "
                + str(mri_img.shape)
                + " (actual: "
                + str(shape)
                + ")"
            )
    """


# +
def is_int(val):
    if type(val) == int:
        return True
    else:
        if val.is_integer():
            return True
        else:
            return False


def preprocess(
    env: aneurysm_utils.Environment, mri_imgs: list, preprocessing_params: dict
) -> list:
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

    return mri_imgs

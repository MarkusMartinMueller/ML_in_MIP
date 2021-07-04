import glob
import os
from typing import List, Optional, Union
from scipy.ndimage import zoom
import nibabel as nib
import nilearn
import numpy as np
import pyvista as pv
import pandas as pd
import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from monai.transforms import Spacing

import aneurysm_utils

DF_DICT = {"mask": "Path Mask", "vessel": "Path Vessel", "rupture risk": "Rupture Status", "labeled":"Path Labeled Mask"}

def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True, resample_dim=(1.5, 1.5, 1.5),resample_size=None,order=2):
    """Load a 3D array from a NIFTI file."""
    nifti = nib.load(file_path)
    struct_arr = nifti.get_data().astype("<f4")
    #print(struct_arr.shape)
    #values, count = np.unique(struct_arr, return_counts=True)
    #print("Original", count, (count[1]/count[0]))

    if resample_dim is not None:
        struct_arr = np.expand_dims(struct_arr, axis=0)
        ###To Do: Resampling method: Lanczos
        spacing = Spacing(pixdim=resample_dim)
        struct_arr = spacing(struct_arr, nifti.affine)[0]
        struct_arr = np.squeeze(struct_arr, axis=0)
    elif resample_size is not None:
        shape=struct_arr.shape
        zoom_factors= [resample_size[0]/shape[0],resample_size[1]/shape[1],resample_size[2]/shape[2]]
        struct_arr=zoom(struct_arr,zoom_factors,order=order)
    #print(struct_arr.shape)


    # struct_arr = np.array(nib.load(file_path).get_data().astype("<f4"))  # TODO:
    # nilearn.image.smooth_img(row["path"], fwhm=3).get_data().astype("<f4")
    # struct_arr = nib.load(file_path).get_data().astype("<f4")
    # np.array(nib.load(file_path).get_data().astype("<f4"))
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr



def save_nifti(file_path, struct_arr):
    """Save a 3D array to a NIFTI file."""
    img = nib.Nifti1Image(struct_arr, np.eye(4))
    nib.save(img, file_path)

def get_orig_images(
    env: aneurysm_utils.Environment,
    df: pd.DataFrame
):
    mri_imgs = []
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        nifti_orig = nib.load(row["Path Orig"])
        mri_imgs.append(nifti_orig)
    return mri_imgs


def get_case_images(
    env: aneurysm_utils.Environment,
    df: pd.DataFrame,
    case: str,
    mri_data_selection: str = "unprocessed",
    mesh: bool = False,
    resample_voxel_dim: tuple = None,
    ):

    files = {
        "Orig": "_orig.nii.gz", 
        "Vessel": "_vessel.nii.gz", 
        "Mask": "_masks.nii.gz", 
        "Labeled Mask": "_labeledMasks.nii.gz",
        "Vessel Mesh": "_vessel.stl", 
        "Aneurysm Mesh": ".stl",
            }
    img_dict = {}
    img_dict["case"]=case
    for name, file in files.items():
        if ("Mesh" in name):
            if mesh:
                count = int(df.loc[df['Case'] == case].get("Aneurysm Count"))
                try: 
                    path = os.path.join(env.datasets_folder, mri_data_selection, f'{case}{file}')
                    img_dict[name] = pv.read(path)
                except FileNotFoundError:
                    names = "ABCD"
                    for i in names[:count]:
                        path = os.path.join(env.datasets_folder, mri_data_selection, f'{case}_{i}{file}')
                        img_dict[name + " " + i] = pv.read(path)
        else:
            path = os.path.join(env.datasets_folder, mri_data_selection, f'{case}{file}')
            img_dict[name + " nii"] = nib.load(path)
            img_dict[name + " struct_arr"] = load_nifti(path, resample_dim=resample_voxel_dim)
    
    return img_dict

def load_mri_images(
    env: aneurysm_utils.Environment,
    df: pd.DataFrame,
    prediction: str = "mask",
    mask: str = None,
    case_list: List[str] = None,
    resample_voxel_dim: tuple = (1.5, 1.5, 1.5),
    resample_size: tuple=None,
    order:int = 2
):
    """
    Load MRI images for a given dataframe.

    Args:
        env: Environment for data organization.
        df_ppmi: PPMI + MRI dataframe.
        prediction: Selected column to be used as label. Possible values: `mask`, `vessel`, `rupture risk`.

    Returns:
        mri_imgs (ndarray): List of loaded MRI images.
        labels (ndarray): List of labels corresponding to the MRI images.
    """
    assert prediction in ["mask", "vessel", "rupture risk","labeled"]

    if case_list:
        df = df.loc[df["Case"].isin(case_list)]

    mri_imgs = []
    labels = []
    participants = []

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # nifti_orig = nib.load(row["Path Orig"])
        nifti_orig = load_nifti(row["Path Orig"], resample_dim=resample_voxel_dim,resample_size=resample_size,order=order)
        if prediction in ["mask", "vessel","labeled"]:
            # nifti_mask = nib.load(row[DF_DICT[prediction]])
            nifti_mask = load_nifti(row[DF_DICT[prediction]], resample_dim=resample_voxel_dim,resample_size=resample_size,order=order)
            labels.append(np.rint(nifti_mask))
            #values, count = np.unique(np.rint(nifti_mask), return_counts=True)
            #print(count, (count[1]/count[0]))
        else:
            labels.append(row[DF_DICT[prediction]])
            nifti_labeled_mask = load_nifti(row["Path Labeled Mask"], resample_dim=resample_voxel_dim,resample_size=resample_size,order=order)
            nifti_labeled_mask[nifti_labeled_mask != row["Labeled Mask Index"]] = 0
            # TODO: Add resample here
            nifti_orig *= nifti_labeled_mask

        mri_imgs.append(nifti_orig)
        participants.append(row["Case"])

    return mri_imgs, labels, participants


def load_aneurysm_dataset(
    env: aneurysm_utils.Environment,
    mri_data_selection: str = "unprocessed",
    random_state: int = 0,
    limit: Optional[int] = None,
    prediction: str = "mask",
) -> pd.DataFrame:
    """
    Load, merge, and filter the Meta & MRI data.

    Args:
        env: Environment for data organization.
        ppmi_user: Username to access the ppmi dataset.
        ppmi_pwd: Password to access the ppmi dataset.
        mri_data_selection: Selection of MRI image data. Possible values: `unprocessed`, `brainprep`, `smriprep`.
        random_state: Random state for all sampeling methods.
        limit: Maximum number of samples to load.

    Returns:
        df_ppmi (DataFrame): Dataframe containing the MRI metadata.
    """
    unprocessed_data_path = os.path.abspath(os.path.join(env.datasets_folder, mri_data_selection))
    df = pd.read_excel(
        os.path.join(env.datasets_folder, "Training-RuptureInfo.xlsx"), engine='openpyxl', header=1
    )
    df = df.rename(columns={
        'Rupture Status (1=ruptured, 0=not ruptured, n/a=no rupture information)': 'Rupture Status', 
        "Age " : "Age"
    })
    df["Age Bin"] = pd.cut(
        df["Age"], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    )
    df["Aneurysm Count"] = df.groupby('Angiography Data')['Angiography Data'].transform('count')
    if prediction != "rupture risk":
        df = df.drop_duplicates(subset=['Angiography Data'])
    df["Case"] = df["Angiography Data"].apply(lambda x: x[:-12])
    df["Path Orig"] = df["Case"].apply(lambda x: os.path.join(unprocessed_data_path, f'{x}_orig.nii.gz'))
    df["Path Mask"] = df["Case"].apply(lambda x: os.path.join(unprocessed_data_path, f'{x}_masks.nii.gz'))
    df["Path Vessel"] = df["Case"].apply(lambda x: os.path.join(unprocessed_data_path, f'{x}_vessel.nii.gz'))
    df["Path Labeled Mask"] = df["Case"].apply(lambda x: os.path.join(unprocessed_data_path, f'{x}_labeledMasks.nii.gz'))

    # TODO: Add stl files

    if limit:
        df = df.sample(limit, random_state=random_state)

    return df


def split_mri_images(
    env: aneurysm_utils.Environment,
    df: pd.DataFrame,
    prediction: str = "mask",
    test_size: float = 0.10,
    validation_size: float = 0.10,
    encode_labels: bool = False,
    balance_data: bool = False,
    random_state: int = 0,
    print_stats: bool = True,
    resample_voxel_dim: tuple = (1.5, 1.5, 1.5),
    resample_size: tuple=None,
    order:int = 2
) -> Union[tuple, tuple, tuple, preprocessing.LabelEncoder]:
    """
    Load mri images for provided dataset and split into train, test, and validate.

    Args:
        env: Environment for data organization.
        df: Dataframe that contains the dataset information and path to images.
        prediction: Selected mask or column to be used as label.
        test_size: Percentage of data to use for the test dataset.
        validation_size: Percentage of data to use for the validation dataset.
        encode_labels: If `True`, all labels will be encoded to numeric values.
        balance_data: If `True`, data will be balanced based on the selected label.
        random_state: Random state for all sampeling methods.
        print_stats: If `True`, statistics about the dataset split are printed.

    Returns:
        train_data (tuple): Tuple of MRI images and labels for training.
        test_data (tuple): Tuple of MRI images and labels for testing.
        val_data (tuple): Tuple of MRI images and labels for validation.
        label_encoder (LabelEncoder): Label encoder used for encoding labels.
    """

    df = df.loc[df[DF_DICT[prediction]].notnull()]

    if balance_data:
        len_before = len(df)
        # Remove unused categories in label
        try:
            df[prediction] = df[
                [DF_DICT[prediction]]
            ].cat.remove_unused_categories()
        except Exception:
            pass
        df_grouped = df.groupby(DF_DICT[prediction])
        # balance data
        df = df_grouped.apply(
            lambda x: x.sample(
                df_grouped.size().min(), random_state=random_state
            )
        ).reset_index(drop=True)
        print("Removed for balancing: " + str(len_before - len(df)))

    label_encoder = preprocessing.LabelEncoder()

    if encode_labels:
        label_encoder.fit(df[DF_DICT[prediction]])
        df[prediction] = label_encoder.transform(df[DF_DICT[prediction]])

    # Split train and test set
    print(len(df))
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    # Split train and validation set
    print(len(df_train))
    df_train, df_validation = train_test_split(
        df_train,
        test_size=(validation_size / (1 - test_size)),
        shuffle=True,
        random_state=random_state,
    )

    if print_stats:
        print_df_stats(
            df, df_train, df_validation, df_test, label_encoder, prediction
        )

    return (
        load_mri_images(env, df_train, prediction, resample_voxel_dim=resample_voxel_dim,resample_size=resample_size,order=order)[:2],
        load_mri_images(env, df_test, prediction, resample_voxel_dim=resample_voxel_dim,resample_size=resample_size,order=order)[:2],
        load_mri_images(env, df_validation, prediction, resample_voxel_dim=resample_voxel_dim,resample_size=resample_size,order=order)[:2],
        label_encoder
    )


# Helpers

def get_img_stats(struct_arr, image_name):
    metrics = {}
    metrics["min_float"] = struct_arr.min()
    metrics["max_float"] = struct_arr.max()
    metrics["mean_float"] = struct_arr.mean()
    metrics["non_zero_mean"] = struct_arr[np.nonzero(struct_arr)].mean()
    metrics["var_float"] = struct_arr.var()
    metrics["std_float"] = struct_arr.std()
    metrics["non_zero"] = np.count_nonzero(struct_arr)
    metrics["image_name"] = image_name
    return metrics


def print_img_stats(env, df, case_list=None, mri_imgs=None):
    """Prints statistics about the image inlcuding vessel and aneurysm"""
    img_metrics = []
    if case_list:
        df = df.loc[df["Case"].isin(case_list)]
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # Get all images
        img_dict = get_case_images(env, df, row["Case"], mesh=True)

        # Analyse original img
        struct_arr = img_dict["Orig struct_arr"]
        if mri_imgs:
            struct_arr = mri_imgs[row["Case"]]
        img_metrics.append(get_img_stats(struct_arr, "Orig"))

        # Analyse only aneurysm part
        aneurysm_struct_arr = struct_arr * img_dict["Mask struct_arr"]
        img_metrics.append(get_img_stats(aneurysm_struct_arr, "Aneurysm"))

        # Analyse only vessel part
        vessel_struct_arr = struct_arr * img_dict["Vessel struct_arr"]
        img_metrics.append(get_img_stats(vessel_struct_arr, "Vessel"))

        # Analyse vessel part without aneurysm
        vessel_struct_arr = vessel_struct_arr * (1 - img_dict["Mask struct_arr"])
        img_metrics.append(get_img_stats(vessel_struct_arr, "Vessel No Aneurysm"))

        # Analyse non aneurysm part
        struct_arr = struct_arr * (1 - img_dict["Mask struct_arr"])
        img_metrics.append(get_img_stats(struct_arr, "No Aneurysm"))

    return pd.DataFrame(img_metrics).groupby("image_name").describe().loc[
        :, (slice(None), ["min", "max", "mean", "std"])
    ].astype("object").T



def print_df_stats(df, df_train, df_val, df_test, label_encoder, prediction):
    """Print some statistics of the splitted dataset."""
    try:
        labels = list(label_encoder.classes_)
    except AttributeError:
        labels = []
    headers = ["Images"]
    for label in labels:
        headers.append("-> " + str(label))

    def get_stats(df):
        lenghts = [len(df)]
        for label in range(len(labels)):
            df_label = df[df[DF_DICT[prediction]] == label]
            lenghts.append(
                str(len(df_label))
                + " ("
                + str(round((len(df_label) / len(df)), 2))
                + ")"
            )
        return lenghts

    stats = []
    stats.append(["All"] + get_stats(df))
    stats.append(["Train"] + get_stats(df_train))
    stats.append(["Val"] + get_stats(df_val))
    stats.append(["Test"] + get_stats(df_test))

    print(tabulate(stats, headers=headers))
    print()




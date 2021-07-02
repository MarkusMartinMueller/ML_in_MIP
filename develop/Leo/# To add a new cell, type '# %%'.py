# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # DBSCAN Post Processing
# 
# **In this notebook:**
# 
# * Test DBSCAN for post processing
# %% [markdown]
# ## Dependencies
# Install, load, and initialize all required dependencies for this experiment.
# 
# ### Install Dependencies

# %%
#It should be possible to run the notebook independent of anything else. 
# If dependency cannot be installed via pip, either:
# - download & install it via %%bash
# - atleast mention those dependecies in this section

import sys
get_ipython().system('{sys.executable} -m pip install -q -e ../../utils/')

# %% [markdown]
# # System libraries

# %%
from __future__ import absolute_import, division, print_function
import logging, os, sys

# Enable logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)

# Re-import packages if they change
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Recursion Depth
import sys
sys.setrecursionlimit(10000)

# Intialize tqdm to always use the notebook progress bar
import tqdm
tqdm.tqdm = tqdm.tqdm_notebook

# Third-party libraries
import comet_ml
import numpy as np
import pandas as pd
import nilearn.plotting as nip
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import collections
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (12,6)
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'  # adapt plots for retina displays")
import git


# Project utils

import aneurysm_utils
from aneurysm_utils import evaluation, training


# %%
os.getcwd()

# %% [markdown]
# ### Initialize Environment

# %%
env = aneurysm_utils.Environment(project="ML_IN_MIP", root_folder="/workspace/")
env.cached_data["comet_key"] = "EGrR4luSis87yhHbs2rEaqAWs" 
env.print_info()

# %% [markdown]
# ## Load Data
# Download, explore, and prepare all required data for the experiment in this section.

# %%
dataset_params = {
    "prediction": "LabeledMask",
    "mri_data_selection": "unprocessed", 
    "balance_data": False,
    "seed": 1,
    "resample_voxel_dim": None
}

preprocessing_params = {
    'min_max_normalize': None,
    'mean_std_normalize': False,
    'smooth_img': False, # can contain a number: smoothing factor
}

# %% [markdown]
# ### Load Meta Data

# %%
from aneurysm_utils.data_collection import load_aneurysm_dataset

df = load_aneurysm_dataset(
    env,
    mri_data_selection=dataset_params["mri_data_selection"],
    random_state=dataset_params["seed"]
)
df.head()


# %%
from aneurysm_utils import data_collection 


# %%
img_dict = data_collection.get_case_images(env, df, "A130_R", mesh=False, resample_voxel_dim=dataset_params["resample_voxel_dim"])


# %%
img_dict.keys()


# %%
nip.view_img(
    img_dict["Mask nii"], 
    symmetric_cmap=False,
    cmap="Greys_r",
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False,
)


# %%
np.unique(img_dict["Labeled Mask struct_arr"])


# %%
from sklearn.cluster import DBSCAN
from typing import List

def dbscan(mri_images:List(np.array)):
    for image in mri_images:
        db = DBSCAN(eps=1, min_samples=5).fit()
    return mri_images

images =[np.array(np.where(img_dict["Mask struct_arr"]==1)).T]
dbscan(images)

# %%
np.unique(db.labels_)


# %%
dbscan_mask = np.zeros((img_dict["Mask struct_arr"].shape))
for i, label in zip(np.array(np.where(img_dict["Mask struct_arr"]==1)).T, db.labels_):
    dbscan_mask[i[0], i[1], i[2]] = label+1


# %%
nip.view_img(
    nib.Nifti1Image(dbscan_mask, np.eye(4)), 
    symmetric_cmap=False,
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False,
)


# %%
nip.view_img(
    img_dict["Labeled Mask nii"], 
    symmetric_cmap=False,
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False,
)


# %%
dbscan_sets = []
mask_sets = []
for array in [dbscan_mask, mask]:
    for i in np.unique(dbscan_mask).tolist():
        aneurysm_index = np.array(np.where(dbscan_mask==(i-1))).T
        x_hashable = map(tuple, aneurysm_index)
        y = set(x_hashable)
        dbscan_sets.append(y)
        print(y)


# %%




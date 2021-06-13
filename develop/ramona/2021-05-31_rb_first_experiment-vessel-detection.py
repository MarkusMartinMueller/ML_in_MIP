#!/usr/bin/env python
# coding: utf-8

# # Experiment Template
# 
# 
# **In this notebook:**
# 
# * Load original mri data + vessel mask
# * Resample Images to 1.5 mm Voxelsize
# * Filter images based on size
# * Train network to predict vessel mask
# * Evaluate vessel mask
# 
# **Todo:**
# * Change prediction from mask to vessel
# * Check percentage of 1s in resampled mask
# * Write evaluation
# * Try out different batch_sizes

# ## Dependencies
# Install, load, and initialize all required dependencies for this experiment.
# 
# ### Install Dependencies

# In[ ]:


#It should be possible to run the notebook independent of anything else. 
# If dependency cannot be installed via pip, either:
# - download & install it via %%bash
# - atleast mention those dependecies in this section

#get_ipython().system('pip install -q -e ../utils/')



# ### Import Dependencies

# # System libraries

# In[ ]:


from __future__ import absolute_import, division, print_function
import logging, os, sys

# Enable logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)

# Re-import packages if they change
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

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
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (12,6)
#get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'  # adapt plots for retina displays")
import git


# Project utils
import aneurysm_utils
from aneurysm_utils import evaluation, training


# ### Initialize Environment

# In[ ]:


env = aneurysm_utils.Environment(project="ML_in_MIP", root_folder="/workspace/")
env.cached_data["comet_key"] = "EGrR4luSis87yhHbs2rEaqAWs" 
env.print_info()


# ## Load Data
# Download, explore, and prepare all required data for the experiment in this section.

# In[ ]:


dataset_params = {
    "prediction": "vessel",
    "mri_data_selection": "unprocessed",
    "balance_data": False,
    "seed": 1,
    "resample_voxel_dim": (1.5, 1.5, 1.5)
}


preprocessing_params = {
    'min_max_normalize': True,
    'mean_std_normalize': False,
    'smooth_img': False, # can contain a number: smoothing factor
}


# ### Load Meta Data

# In[ ]:


from aneurysm_utils.data_collection import load_aneurysm_dataset

df = load_aneurysm_dataset(
    env,
    mri_data_selection=dataset_params["mri_data_selection"],
    random_state=dataset_params["seed"]
)
df.head()


# ### Load & Split MRI Data

# In[ ]:


# Load MRI images and split into train, test, and validation
from aneurysm_utils.data_collection import split_mri_images
#case_list = ["A001", "A130_R", "A005"]
#df = df.loc[df["Case"].isin(case_list)]

train_data, test_data, val_data, _ = split_mri_images(
    env, 
    df, 
    prediction=dataset_params["prediction"], 
    encode_labels=False,
    random_state=dataset_params["seed"],
    balance_data=dataset_params["balance_data"]
)

mri_imgs_train, labels_train = train_data
mri_imgs_test, labels_test = test_data
mri_imgs_val, labels_val = val_data


# In[ ]:


from aneurysm_utils import preprocessing

preprocessing.check_mri_shapes(mri_imgs_train)


# ## Transform & Preprocess Data

# In[ ]:


from aneurysm_utils import preprocessing

size_of_train = len(mri_imgs_train)
size_of_test = len(mri_imgs_test)
size_of_val = len(mri_imgs_val)

# preprocess all lists as one to have a working mean_std_normalization
mri_imgs = mri_imgs_train + mri_imgs_test + mri_imgs_val
mri_imgs = preprocessing.preprocess(env, mri_imgs, preprocessing_params)

mri_imgs_train = mri_imgs[:size_of_train]
mri_imgs_train = [train for train in mri_imgs_train]
mri_imgs_test = mri_imgs[size_of_train : size_of_train + size_of_test]
mri_imgs_test = [test for test in mri_imgs_test]
mri_imgs_val = mri_imgs[size_of_train + size_of_test :]
mri_imgs_val = [val for val in mri_imgs_val]

# preprocess mask
x, y, h = labels_train[0].shape
labels_train = [label_train for label_train in labels_train]
labels_test = [label_test for label_test in labels_test]
labels_val = [label_val for label_val in labels_val]
# flatten


# In[ ]:


# 32, 32, 32 - funktion
train_index = [i for i, e in enumerate(mri_imgs_train) if e.shape != (93, 93, 80)]
mri_imgs_train = [i[1:, 1:, :] for j, i in enumerate(mri_imgs_train) if j not in train_index]
labels_train = [i[1:, 1:, :] for j, i in enumerate(labels_train) if j not in train_index]

test_index = [i for i, e in enumerate(mri_imgs_test) if e.shape != (93, 93, 80)]
mri_imgs_test = [i[1:, 1:, :] for j, i in enumerate(mri_imgs_test) if j not in test_index]
labels_test = [i[1:, 1:, :] for j, i in enumerate(labels_test) if j not in test_index]

val_index = [i for i, e in enumerate(mri_imgs_val) if e.shape != (93, 93, 80)]
mri_imgs_val = [i[1:, 1:, :] for j, i in enumerate(mri_imgs_val) if j not in val_index]
labels_val = [i[1:, 1:, :] for j, i in enumerate(labels_val) if j not in val_index]

mri_imgs_train[0].shape
preprocessing.check_mri_shapes(mri_imgs_train)
print(np.unique(labels_val[0], return_counts=True))


# ### Optional: View image

# In[ ]:


idx = 0
nip.view_img(
    nib.Nifti1Image(mri_imgs_train[0], np.eye(4)),
    symmetric_cmap=False,
    cmap="Greys_r",
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False
)


# In[ ]:


evaluation.plot_slices(mri_imgs_train[0])


# ## Train Model
# Implementation, configuration, and evaluation of the experiment.
# 
# ### Train Deep Model 3D data

# In[ ]:


artifacts = {
    "train_data": (mri_imgs_train, labels_train),
    "val_data": (mri_imgs_val, labels_val),
    "test_data": (mri_imgs_test, labels_test)
}

# Define parameter configuration for experiment run
params = {
    "batch_size": 3,
    "epochs": 1,
    "learning_rate": 5.0e-3, # 3e-04, 1.0E-5
    "es_patience": None, # None = deactivate early stopping
    "weight_decay": 0.001, # 1e-3
    "model_name": 'SimpleCNN3D',
    "optimizer_momentum": 0.9,
    "optimizer":'Adam',
    "criterion": "CrossEntropyLoss", 
    "criterion_weights": [1.0, 100.0], # [1.75, 1.0],
    "sampler": None,   #'ImbalancedDatasetSampler2',
    "shuffle_train_set": True,
    "scheduler": "ReduceLROnPlateau", # "ReduceLROnPlateau",
    "save_models": False,
    "debug": True,
}

params.update(dataset_params)
params.update(preprocessing_params)

# data augmentation


# In[ ]:


# Run experiment and sync all metadata
exp = env.create_experiment(
    params["prediction"] + "-pytorch-" + params["model_name"],
    comet_ml.Experiment(
        env.cached_data["comet_key"],
        project_name=env.project + "-" + params["prediction"],
        disabled=params["debug"],
    ),
)
exp.run(training.train_pytorch_model, params, artifacts)


# ## Evaluate Model
# 
# Do evaluation, e.g. visualizations  

# In[ ]:


from aneurysm_utils.utils.pytorch_utils import predict


# In[ ]:


model = exp.artifacts["model"]


# In[ ]:


predictions = predict(model, mri_imgs_val,apply_softmax=False )


# In[ ]:



idx = 0
nip.view_img(
    nib.Nifti1Image(predictions[0][0], np.eye(4)),
    symmetric_cmap=False,
    cmap="Greys_r",
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False
)


# In[ ]:


idx = 0
nip.view_img(
    nib.Nifti1Image(labels_val[0], np.eye(4)),
    symmetric_cmap=False,
    cmap="Greys_r",
    bg_img=False,
    black_bg=True,
    threshold=1e-03, 
    draw_cross=False
)


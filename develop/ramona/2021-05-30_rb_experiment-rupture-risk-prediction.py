#!/usr/bin/env python
# coding: utf-8
# %%
# # First Rupture Risk Experiment <a class="tocSkip">

# In this notebook, I wanna test a simple CNN model to pretict the rupture risk of an aneurysm.
#
# TODO: estimate risk of every aneurysm in
#
# **In this notebook:**
#
# * Describe your notbeook here in a few bullet points, e.g.:
# * Method xyz on dataset abc --> Key insight: xyz works pretty well
# * Modification zyx --> Dead end
#
# **Todo:**
#
# * List all todos that are related to this notebook here, e.g.:
# * Apply xyz to another dataset
#
# This could be some more general information on method xyz (e.g. a link to a paper).
#
# _Please use a Python 3 kernel for the notebook_

# ## Dependencies
# Install, load, and initialize all required dependencies for this experiment.

# ### Install Dependencies

# %%


# It should be possible to run the notebook independent of anything else. 
# If dependency cannot be installed via pip, either:
# - download & install it via %%bash
# - atleast mention those dependecies in this section

#get_ipython().system('pip install -q -e ../utils/')


# ### Import Dependencies

# %%
# System libraries
# from __future__ import absolute_import, division, print_function
import logging, os, sys

# Enable logging
logging.basicConfig(
    format="[%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stdout
)

# Re-import packages if they change
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# Intialize tqdm to always use the notebook progress bar
import tqdm

tqdm.tqdm = tqdm.tqdm_notebook

# Third-party libraries
import numpy as np
import pandas as pd
import nilearn.plotting as nip
import matplotlib.pyplot as plt
import nibabel as nib

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (12, 6)
# get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'  # adapt plots for retina displays")
import git
import comet_ml

# Project utils
import aneurysm_utils
from aneurysm_utils import evaluation, training


# ### Initialize Environment

# %%


env = aneurysm_utils.Environment(project="ML_in_MIP", root_folder="/workspace/")
env.cached_data["comet_key"] = "EGrR4luSis87yhHbs2rEaqAWs" 
env.print_info()


# ## Load Data
# Download, explore, and prepare all required data for the experiment in this section.

# %%


dataset_params = {
    "prediction": "rupture risk",
    "mri_data_selection": "unprocessed", 
    "balance_data": False,
    "seed": 1,
}

preprocessing_params = {
    'min_max_normalize': True,
    'mean_std_normalize': False,
    'smooth_img': False, # can contain a number: smoothing factor
}


# ### Load Meta Data

# %%


from aneurysm_utils.data_collection import load_aneurysm_dataset

df = load_aneurysm_dataset(
    env,
    mri_data_selection=dataset_params["mri_data_selection"],
    random_state=dataset_params["seed"],
    prediction=dataset_params["prediction"],
)
df.head()


# ### Load & Split MRI Data

# %%


# Load MRI images and split into train, test, and validation
from aneurysm_utils.data_collection import split_mri_images
case_list = ["A001", "A130_R", "A005"]
df = df.loc[df["Case"].isin(case_list)]

train_data, test_data, val_data, _ = split_mri_images(
    env, 
    df, 
    prediction=dataset_params["prediction"], 
    encode_labels=True,
    random_state=dataset_params["seed"],
    balance_data=dataset_params["balance_data"]
)

mri_imgs_train, labels_train = train_data
mri_imgs_test, labels_test = test_data
mri_imgs_val, labels_val = val_data


# %%


from aneurysm_utils import preprocessing

preprocessing.check_mri_shapes(mri_imgs_train)


# %%



train_index = [i for i, e in enumerate(mri_imgs_train) if e.shape != (256, 256, 220)]
mri_imgs_train = [i for j, i in enumerate(mri_imgs_train) if j not in train_index]
labels_train = [i for j, i in enumerate(labels_train) if j not in train_index]

test_index = [i for i, e in enumerate(mri_imgs_test) if e.shape != (256, 256, 220)]
mri_imgs_test = [i for j, i in enumerate(mri_imgs_test) if j not in test_index]
labels_test = [i for j, i in enumerate(labels_test) if j not in test_index]

val_index = [i for i, e in enumerate(mri_imgs_val) if e.shape != (256, 256, 220)]
mri_imgs_val = [i for j, i in enumerate(mri_imgs_val) if j not in val_index]
labels_val = [i for j, i in enumerate(labels_val) if j not in val_index]


# ## Transform & Preprocess Data

# %%


from aneurysm_utils import preprocessing

size_of_train = len(mri_imgs_train)
size_of_test = len(mri_imgs_test)
size_of_val = len(mri_imgs_val)

# preprocess all lists as one to have a working mean_std_normalization
mri_imgs = mri_imgs_train + mri_imgs_test + mri_imgs_val
mri_imgs = preprocessing.preprocess(env, mri_imgs, preprocessing_params)

mri_imgs_train = mri_imgs[:size_of_train]
mri_imgs_train = [train[50:140,50:150,40:130] for train in mri_imgs_train]
mri_imgs_test = mri_imgs[size_of_train : size_of_train + size_of_test]
mri_imgs_test = [test[50:140,50:150,40:130] for test in mri_imgs_test]
mri_imgs_val = mri_imgs[size_of_train + size_of_test :]
mri_imgs_val = [val[50:140,50:150,40:130] for val in mri_imgs_val]


# %%


mri_imgs_train[0].shape


# ### Optional: View image
# 

# %%


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


# %%


evaluation.plot_slices(mri_imgs_train[0])


# ## Train Model
# Implementation, configuration, and evaluation of the experiment.

# ### Train Deep Model 3D data

# %%


artifacts = {
    "train_data": (mri_imgs_train, labels_train),
    "val_data": (mri_imgs_val, labels_val),
    "test_data": (mri_imgs_test, labels_test)
}

# Define parameter configuration for experiment run
params = {
    # "seed": 1, moved to dataset params
    "training_size": None,  # None=all
    "val_size": None,   # None=all
    "batch_size": 3,
    "epochs": 250,
    "learning_rate": 5.0e-5, # 3e-04, 1.0E-5
    "es_patience": 75, # None = deactivate early stopping
    "weight_decay": 0.001, # 1e-3
    "model_name": 'CNN3DTutorial', # "resnet", "preresnet", "wideresnet", "densenet", "simpleCNN", "ClassificationModel3D", "CNN3DSoftmax", "CNN3DMoboehle", "CNN3DTutorial", "LinearModel3D",
    #"model_depth": 10, # 10
    #"resnet_shortcut": 'B',
    "optimizer_momentum": 0.9,
    "optimizer":'Adam',
    "criterion": 'CrossEntropyLoss',
    "criterion_weights": False, # [1.75, 1.0],
    "sampler": None,   #'ImbalancedDatasetSampler2',
    "shuffle_train_set": True,
    "scheduler": None, # "ReduceLROnPlateau",
    "save_models": False,
    "debug": True,
    "dropout": 0.4,
    "dropout2": 0.2
    #"pretrain_path": env.get_file("models/resnet_10_23dataset.pth"),
    #"train_pretrain": True,
    #"new_layer_name": ["fc"]
}

params.update(dataset_params)
params.update(preprocessing_params)

# data augmentation
# use dropout
# Cross-validation


# %%


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

# %%


# Do evaluation, e.g. visualizations  


import os
import json
import multiprocessing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.animation
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from scipy.spatial import distance_matrix
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from sklearn.metrics import jaccard_score
from skimage.metrics import hausdorff_distance
from scipy.stats import pearsonr
from aneurysm_utils.preprocessing import resize_mri
from aneurysm_utils.environment import Environment
from collections import defaultdict
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import MinMaxScaler
#import open3d

def evaluate_model(
    y_true: list, y_pred: list, segmentation: bool = None, prefix: str = None
) -> dict:
    metrics = {}

    if segmentation:
        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()

    if not prefix:
        prefix = ""
    else:
        prefix = prefix + "_"
    
    metrics[prefix + "accuracy"] = sk_metrics.accuracy_score(y_true, y_pred)
    metrics[prefix + "bal_acc"] = sk_metrics.balanced_accuracy_score(y_true, y_pred)
    try:
        metrics[prefix + "precision"] = sk_metrics.precision_score(y_true, y_pred)
        metrics[prefix + "recall"] = sk_metrics.recall_score(y_true, y_pred)
        metrics[prefix + "spec"] = sk_metrics.recall_score(y_true, y_pred, pos_label=0)
        metrics[prefix + "sen"] = sk_metrics.recall_score(y_true, y_pred, pos_label=1)
        metrics[prefix + "f1"] = sk_metrics.f1_score(y_true, y_pred)
    except Exception:
        print(
            "precision/recall/spec/sen/f1 are not supported for non-binary classification."
        )

    print("Accuracy (" + prefix + "): " + str(metrics[prefix + "accuracy"]))
    print("Balanced Accuracy (" + prefix + "): " + str(metrics[prefix + "bal_acc"]))
    print(sk_metrics.classification_report(y_true, y_pred))

    return metrics


# Transparent colormap (alpha to red), that is used for plotting an overlay.
# See https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
alpha_to_red_cmap = np.zeros((256, 4))
alpha_to_red_cmap[:, 0] = 0.8
alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)  # cmap.N-20)  # alpha values
alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)

red_to_alpha_cmap = np.zeros((256, 4))
red_to_alpha_cmap[:, 0] = 0.8
red_to_alpha_cmap[:, -1] = np.linspace(1, 0, 256)  # cmap.N-20)  # alpha values
red_to_alpha_cmap = mpl.colors.ListedColormap(red_to_alpha_cmap)


def animate_slices(
    struct_arr,
    overlay=None,
    axis=0,
    reverse_direction=False,
    interval=40,
    vmin=None,
    vmax=None,
    overlay_vmin=None,
    overlay_vmax=None,
):
    """
    Create a matplotlib animation that moves through a 3D image along a specified axis.
    """

    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()

    fig, ax = plt.subplots()
    axis_label = ["x", "y", "z"][axis]

    # TODO: If I select slice 50 here at the beginning, the plots look different.
    im = ax.imshow(
        np.take(struct_arr, 0, axis=axis),
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
        interpolation=None,
        animated=True,
    )
    if overlay is not None:
        im_overlay = ax.imshow(
            np.take(overlay, 0, axis=axis),
            vmin=overlay_vmin,
            vmax=overlay_vmax,
            cmap=alpha_to_red_cmap,
            interpolation=None,
            animated=True,
        )
    text = ax.text(
        0.03,
        0.97,
        "{}={}".format(axis_label, 0),
        color="white",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.axis("off")

    def update(i):
        im.set_array(np.take(struct_arr, i, axis=axis))
        if overlay is not None:
            im_overlay.set_array(np.take(overlay, i, axis=axis))
        text.set_text("{}={}".format(axis_label, i))
        return im, text

    num_frames = struct_arr.shape[axis]
    if reverse_direction:
        frames = np.arange(num_frames - 1, 0, -1)
    else:
        frames = np.arange(0, num_frames)

    return mpl.animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )


def plot_slices(
    struct_arr,
    num_slices=7,
    cmap="gray",
    vmin=None,
    vmax=None,
    overlay=None,
    overlay_cmap=alpha_to_red_cmap,
    overlay_vmin=None,
    overlay_vmax=None,
):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis

    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`).
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ["x", "y", "z"]):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            # print(axis_label, 'plotting slice', i_slice)

            plt.sca(ax)
            plt.axis("off")
            plt.imshow(
                sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation=None,
            )
            plt.text(
                0.03,
                0.97,
                "{}={}".format(axis_label, i_slice),
                color="white",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
            )

            if overlay is not None:
                plt.imshow(
                    sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90),
                    cmap=overlay_cmap,
                    vmin=overlay_vmin,
                    vmax=overlay_vmax,
                    interpolation=None,
                )

def draw_mask_3d(image:np.array,ax:Axes3D=None,zorder:int=0,markersize:float=0.8,alpha:float=1,c=None):
    """
    Draws all points which are not zero of given image in scatterplot
    
    Parameters
    ----------
    image: where to get mask from
    ax: if given uses this axis object
    zorder: order of points drawn
    markersize: size of points
    alpha: transparency of points
    c: if anything points will be black
    
    """
    fig = plt.figure()
    if ax==None:
        ax = Axes3D(fig)
    else:
        ax=ax
    for cluster in range(1,int(np.unique(image)[-1]+1)):
        if len(np.argwhere(image==cluster))==0:
            print("no aneurysm found")
            continue
        if c==None:
            ax.scatter(np.argwhere(image==cluster).T[0],np.argwhere(image==cluster).T[1],np.argwhere(image==cluster).T[2],s=markersize,alpha=alpha,zorder=zorder)
        else:
            ax.scatter(np.argwhere(image==cluster).T[0],np.argwhere(image==cluster).T[1],np.argwhere(image==cluster).T[2],s=3,alpha=alpha,zorder=zorder,c="black")

def draw_image(image:np.array,ax:Axes3D=None,zorder:int=0,markersize:float=0.8,transparency:bool=True):
    """
    Draws all points which are not zero of given image in scatterplot in colors according to their intensity
    
    Parameters
    ----------
    image: where to get mask from
    ax: if given uses this axis object
    zorder: order of points drawn
    markersize: size of points
    transparency: if true scales transparency with intensity values

    
    """
    fig = plt.figure()
    if ax==None:
        ax = Axes3D(fig)
    else:
        ax=ax
    if transparency:
        alpha= image[image>0]
        alpha = np.where(alpha>0.15,alpha,0.01)
    else:
        alpha=1
    cmap = plt.get_cmap('YlOrRd')
    ax.scatter(np.argwhere(image>0).T[0],np.argwhere(image>0).T[1],np.argwhere(image>0).T[2],s=markersize,alpha=image[image>0],zorder=zorder,c=cmap(image[image>0]))


def draw_bounding_box(candidates,ax:Axes3D=None):
    """
    Draws bounding box of given bounding box dictionary -> see postprocessing function
    
    Parameters
    ----------
    image: list of dictionaries where entry vertices contains the points of the bounding box
    ax: if given uses this axis object


    """
    fig = plt.figure()
    if ax==None:
        ax = Axes3D(fig)
    else:
        ax=ax
    for candidate in candidates:
        Z= candidate["vertices"]
        Z=np.array(Z)

        verts= [(Z[0],Z[1]),(Z[0],Z[2]),(Z[0],Z[3]),(Z[6],Z[1]),(Z[7],Z[1]),(Z[2],Z[5]),
        (Z[2],Z[7]),(Z[3],Z[5]),(Z[3],Z[6]),(Z[4],Z[7]),(Z[4],Z[6]),(Z[4],Z[5])]

        for element in verts:
            x=[element[0][0],element[1][0]]
            y=[element[0][1],element[1][1]]
            z=[element[0][2],element[1][2]]
            ax.plot(x,y,z,c='r',linewidth=2,alpha=1)
    fig.show()





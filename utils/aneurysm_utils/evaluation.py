# +
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

# -
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
import open3d

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

def draw_mask_3d(image:np.array,ax=None,zorder=0,markersize=0.8,alpha=1,limits=(0,0,0)):
    fig = plt.figure()
    if ax==None:
        ax = Axes3D(fig)
        if limits!=(0,0,0):
            ax.xlim3d=limits[0]
            ax.ylim3d=limits[1]
            ax.zlim3d=limits[2]
    else:
        ax=ax
    ax.scatter(np.argwhere(image).T[0],np.argwhere(image).T[1],np.argwhere(image).T[2],s=markersize,alpha=alpha,zorder=zorder)



def draw_bounding_box(candidates,vessel_array:np.array=None,aneurysm_array:np.array=None,limits=(0,0,0)):
    fig = plt.figure()
    ax = Axes3D(fig)
    for candidate in candidates:
        Z= candidate["vertices"]
        Z=np.array(Z)
       
        if limits!=(0,0,0):
            print("setting new limits")
            ax.xlim3d=limits[0]
            ax.ylim3d=limits[1]
            ax.zlim3d=limits[2]
        verts= [(Z[0],Z[1]),(Z[0],Z[2]),(Z[0],Z[3]),(Z[6],Z[1]),(Z[7],Z[1]),(Z[2],Z[5]),
        (Z[2],Z[7]),(Z[3],Z[5]),(Z[3],Z[6]),(Z[4],Z[7]),(Z[4],Z[6]),(Z[4],Z[5])]

        for element in verts:
            x=[element[0][0],element[1][0]]
            y=[element[0][1],element[1][1]]
            z=[element[0][2],element[1][2]]
            ax.plot(x,y,z,c='r',zorder=2,linewidth=2,alpha=1)

    if vessel_array is not None:
        draw_mask_3d(vessel_array,ax,zorder=-1,markersize=3,alpha=0.2)
    if aneurysm_array is not None:
        draw_mask_3d(aneurysm_array,ax,zorder=1,markersize=3,alpha=0.8)
    fig.show()
# +
# ---------------------------- Interpretation methods --------------------------------
# From: https://github.com/jrieke/cnn-interpretability


def sensitivity_analysis(
    model,
    image_tensor,
    target_class=None,
    postprocess="abs",
    apply_softmax=True,
    cuda=False,
    verbose=False,
):
    """
    Perform sensitivity analysis (via backpropagation; Simonyan et al. 2014) to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used
                                                 in Simonyan et al. 2014, `'square'` is used in Montavon et al. 2018.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        appl (None or 'binary' or 'categorical'): Whether the output format of the `model` is binary
                                                         (i.e. one output neuron with sigmoid activation) or categorical
                                                         (i.e. multiple output neurons with softmax activation).
                                                         If `None` (default), infer from the shape of the output.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """
    if postprocess not in [None, "abs", "square"]:
        raise ValueError("postprocess must be None, 'abs' or 'square'")

    # Forward pass.
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(
        image_tensor[None], requires_grad=True
    )  # add dimension to simulate batch
    output = model(X)
    if apply_softmax:
        output = F.softmax(output)

    # Backward pass.
    model.zero_grad()
    output_class = output.max(1)[1].data[0]
    if verbose:
        print(
            "Image was classified as",
            output_class,
            "with probability",
            output.max(1)[0].data[0],
        )
    one_hot_output = torch.zeros(output.size())
    if target_class is None:
        one_hot_output[0, output_class] = 1
    else:
        one_hot_output[0, target_class] = 1
    if cuda:
        one_hot_output = one_hot_output.cuda()
    output.backward(gradient=one_hot_output)

    relevance_map = X.grad.data[0].cpu().numpy()

    # Postprocess the relevance map.
    if postprocess == "abs":  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == "square":  # as in Montavon et al. (2018)
        return relevance_map ** 2
    elif postprocess is None:
        return relevance_map


def guided_backprop(
    model,
    image_tensor,
    target_class=None,
    postprocess="abs",
    apply_softmax=True,
    cuda=False,
    verbose=False,
):
    """
    Perform guided backpropagation (Springenberg et al. 2015) to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Note: The `model` MUST implement any ReLU layers via `torch.nn.ReLU` (i.e. it needs to have an instance
    of torch.nn.ReLU as an attribute). Models that use `torch.nn.functional.relu` instead will not work properly!

    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        postprocess (None or 'abs' or 'square'): The method to postprocess the heatmap with. `'abs'` is used
                                                 in Simonyan et al. 2013, `'square'` is used in Montavon et al. 2018.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """
    layer_to_hook = nn.ReLU

    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    hook_handles = []

    try:
        # Loop through layers, hook up ReLUs with relu_hook_function, store handles to hooks.
        for module in model.children():
            # TODO: Maybe hook on ELU layers as well (or on others?).
            if isinstance(module, layer_to_hook):
                # TODO: Add a warning if no activation layers have been hooked, so that the user does not forget
                #       to invoke the activation via nn.ReLU instead of F.relu.
                if verbose:
                    print("Registered hook for layer:", module)
                hook_handle = module.register_backward_hook(relu_hook_function)
                hook_handles.append(hook_handle)

        # Calculate backprop with modified ReLUs.
        relevance_map = sensitivity_analysis(
            model,
            image_tensor,
            target_class=target_class,
            postprocess=postprocess,
            apply_softmax=apply_softmax,
            cuda=cuda,
            verbose=verbose,
        )

    finally:
        # Remove hooks from model.
        # The finally clause re-raises any possible exceptions.
        if verbose:
            print("Removing {} hook(s)".format(len(hook_handles)))
        for hook_handle in hook_handles:
            hook_handle.remove()
            del hook_handle

    return relevance_map


def occlusion(
    model,
    image_tensor,
    target_class=None,
    size=50,
    stride=25,
    occlusion_value=0,
    apply_softmax=True,
    three_d=None,
    resize=True,
    cuda=False,
    verbose=False,
):
    """
    Perform occlusion (Zeiler & Fergus 2014) to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Note: The current implementation can only handle 2D and 3D images.
    It usually infers the correct image dimensions, otherwise they can be set via the `three_d` parameter.

    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        size (int): The size of the occlusion patch.
        stride (int): The stride with which to move the occlusion patch across the image.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        three_d (boolean): Whether the image is 3 dimensional (e.g. MRI scans).
                           If `None` (default), infer from the shape of `image_tensor`.
        resize (boolean): The output from the occlusion method is usually smaller than the original `image_tensor`.
                          If `True` (default), the output will be resized to fit the original shape (without interpolation).
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """

    # TODO: Try to make this better, i.e. generalize the method to any kind of input.
    if three_d is None:
        three_d = len(image_tensor.shape) == 4  # guess if input image is 3D

    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False)).cpu()
    if apply_softmax:
        output = F.softmax(output)

    output_class = output.max(1)[1].data.numpy()[0]
    if verbose:
        print(
            "Image was classified as",
            output_class,
            "with probability",
            output.max(1)[0].data[0],
        )
    if target_class is None:
        target_class = output_class
    unoccluded_prob = output.data[0, target_class]

    width = image_tensor.shape[1]
    height = image_tensor.shape[2]

    xs = range(0, width, stride)
    ys = range(0, height, stride)

    # TODO: Maybe use torch tensor here.
    if three_d:
        depth = image_tensor.shape[3]
        zs = range(0, depth, stride)
        relevance_map = np.zeros((len(xs), len(ys), len(zs)))
    else:
        relevance_map = np.zeros((len(xs), len(ys)))

    if verbose:
        xs = tqdm(xs, desc="x")
        ys = tqdm(ys, desc="y", leave=False)
        if three_d:
            zs = tqdm(zs, desc="z", leave=False)

    image_tensor_occluded = image_tensor.clone()  # TODO: Check how long this takes.

    if cuda:
        image_tensor_occluded = image_tensor_occluded.cuda()

    for i_x, x in enumerate(xs):
        x_from = max(x - int(size / 2), 0)
        x_to = min(x + int(size / 2), width)

        for i_y, y in enumerate(ys):
            y_from = max(y - int(size / 2), 0)
            y_to = min(y + int(size / 2), height)

            if three_d:
                for i_z, z in enumerate(zs):
                    z_from = max(z - int(size / 2), 0)
                    z_to = min(z + int(size / 2), depth)

                    # if verbose: print('Occluding from x={} to x={} and y={} to y={} and z={} to z={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))

                    image_tensor_occluded.copy_(image_tensor)
                    image_tensor_occluded[
                        :, x_from:x_to, y_from:y_to, z_from:z_to
                    ] = occlusion_value

                    # TODO: Maybe run this batched.
                    output = model(
                        Variable(image_tensor_occluded[None], requires_grad=False)
                    )
                    if apply_softmax:
                        output = F.softmax(output)

                    occluded_prob = output.data[0, target_class]
                    relevance_map[i_x, i_y, i_z] = unoccluded_prob - occluded_prob

            else:
                # if verbose: print('Occluding from x={} to x={} and y={} to y={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))
                image_tensor_occluded.copy_(image_tensor)
                image_tensor_occluded[:, x_from:x_to, y_from:y_to] = occlusion_value

                # TODO: Maybe run this batched.
                output = model(
                    Variable(image_tensor_occluded[None], requires_grad=False)
                )
                if apply_softmax:
                    output = F.softmax(output)

                occluded_prob = output.data[0, target_class]
                relevance_map[i_x, i_y] = unoccluded_prob - ocluded_prob

    relevance_map = np.maximum(relevance_map, 0)

    if resize:
        relevance_map = resize_mri(relevance_map, image_tensor.shape[1:])

    return relevance_map


def area_occlusion(
    model,
    image_tensor,
    area_masks,
    target_class=None,
    occlusion_value=0,
    apply_softmax=True,
    cuda=False,
    verbose=False,
):
    """
    Perform brain area occlusion to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """

    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))
    if apply_softmax:
        output = F.softmax(output)

    output_class = output.max(1)[1].data.cpu().numpy()[0]
    if verbose:
        print(
            "Image was classified as",
            output_class,
            "with probability",
            output.max(1)[0].data[0],
        )
    if target_class is None:
        target_class = output_class
    unoccluded_prob = output.data[0, target_class]

    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()

    for area_mask in tqdm(area_masks) if verbose else area_masks:
        # TODO: Maybe have area_mask as tensor in the first place.
        area_mask = torch.FloatTensor(area_mask)
        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)

        output = model(Variable(image_tensor_occluded[None], requires_grad=False))
        if apply_softmax:
            output = F.softmax(output)

        occluded_prob = output.data[0, target_class]
        relevance_map[area_mask.view(image_tensor.shape) == 1] = (
            unoccluded_prob - occluded_prob
        )

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map


def get_aal_atlas(env: Environment, shape):
    import nilearn
    import aneurysm_utils

    atlas_aal = nilearn.datasets.fetch_atlas_aal(data_dir=env.datasets_folder)
    atlas_map_path = atlas_aal["maps"]
    area_names = atlas_aal["labels"]

    brain_map = aneurysm_utils.data_collection.load_nifti(atlas_map_path)
    brain_areas = np.unique(brain_map)[1:]  # omit background

    area_masks = []
    for area in tqdm(brain_areas):
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = aneurysm_utils.preprocessing.resize_mri(
            area_mask, shape, interpolation=0
        )
        area_masks.append(area_mask)

    # Merge left and right areas.
    merged_area_names = [name[:-2] for name in area_names[:108:2]] + area_names[108:]
    return area_masks, merged_area_names


def get_relevance_per_area(relevance_map, area_masks, area_names, normalize=True):
    relevances = np.zeros(len(area_masks))
    for i, area_mask in enumerate(area_masks):
        relevances[i] = np.sum(relevance_map * area_mask)
    if normalize:
        relevances /= relevances.sum()  # make all areas sum to 1

    # Merge left and right areas.
    merged_relevances = np.concatenate(
        [relevances[:108].reshape(-1, 2).sum(1), relevances[108:]]
    )
    return sorted(zip(area_names, merged_relevances), key=lambda a: a[1], reverse=True)


# ----------------------------------- Averages over datasets ---------------


def average_over_dataset(
    interpretation_method,
    model,
    dataset,
    num_samples=None,
    seed=None,
    show_progress=False,
    **kwargs
):
    """Apply an interpretation method to each sample of a dataset, and average separately over AD and NC samples."""

    if seed is not None:
        np.random.seed(seed)

    # Run once to figure out shape of the relevance map. Cannot be inferred
    # from image shape alone, because some interpretation methods contain
    # a channel dimension and some not.
    struct_arr, label = dataset[0]
    relevance_map = interpretation_method(model, struct_arr, **kwargs)

    avg_relevance_map_AD = np.zeros_like(relevance_map)
    avg_relevance_map_NC = np.zeros_like(relevance_map)

    count_AD = 0
    count_NC = 0

    if num_samples is None:
        idx = range(len(dataset))
    else:
        idx = np.random.choice(len(dataset), num_samples, replace=False)
    if show_progress:
        idx = tqdm(idx)

    for i in idx:
        struct_arr, label = dataset[i]

        relevance_map = interpretation_method(model, struct_arr, **kwargs)

        if label == 1:
            # print(label, 'adding to AD', relevance_map.shape)
            avg_relevance_map_AD += relevance_map  # [0]
            count_AD += 1
        else:
            # print(label, 'adding to NC', relevance_map.shape)
            avg_relevance_map_NC += relevance_map  # [0]
            count_NC += 1

    avg_relevance_map_AD /= count_AD
    avg_relevance_map_NC /= count_NC

    avg_relevance_map_all = (
        avg_relevance_map_AD * count_AD + avg_relevance_map_NC * count_NC
    ) / (count_AD + count_NC)

    return avg_relevance_map_AD, avg_relevance_map_NC, avg_relevance_map_all


# ------------------------------ Distance between heatmaps ----------------------------


def heatmap_distance(a, b):
    """Calculate the Euclidean distance between two n-dimensional arrays."""

    def preprocess(arr):
        """Preprocess an array for use in Euclidean distance."""
        # arr = arr * mask
        arr = arr.flatten()
        arr = arr / arr.sum()  # normalize to sum 1
        # arr = arr.clip(1e-20, None)  # avoid 0 division in KL divergence
        return arr

    a, b = preprocess(a), preprocess(b)

    # Euclidean distance.
    return np.sqrt(np.sum((a - b) ** 2))

    # Wasserstein Distance.
    # return sp.stats.wasserstein_distance(a, b)

    # Symmetric KL Divergence (ill-defined for arrays with 0 values!).
    # return sp.stats.entropy(a, b) + sp.stats.entropy(b, a)

#-------------------------------Postprocessing Evaluation-----------------------------
def evaluate_dbscan(predicted:List[np.array],groundtruth:List[np.array],cases:List["str"]):
    all_scores={}
    for pred,truth,case in zip(predicted,groundtruth,cases):
        all_scores[case]={}
        for cluster in range(1,int(np.unique(pred)[-1])+1):
            indices =list(np.array(np.where(pred==cluster)).T)
            all_scores[case]["predicted_cluster_"+str(cluster)]={}
            for true_cluster in range(1,int(np.unique(truth)[-1])+1):
                true_indices = list(np.array(np.where(truth==true_cluster)).T)

                all_scores[case]["predicted_cluster_"+str(cluster)]["true_cluster_"+str(true_cluster)]=compare_two_list(true_indices,indices)/len(indices)

    return all_scores

def compare_two_list(list_a,list_b):
    list_a= set([tuple(x)for x in list_a])
    list_b= set([tuple(x)for x in list_b])

    return len(list_a.intersection(list_b))


#---------------------------------Scores------------------------------------------------


def coverage(boxobjects:List,labeled_aneurysm_mask:np.array):
    total_score=0
    for box_dict in boxobjects:
            scorebefore=0
            for label in range(1,int(np.unique(labeled_aneurysm_mask)[-1])+1):
                aneurysm = np.array(np.where(labeled_aneurysm_mask==label)).T
                print(len(aneurysm))
                print(len(box_dict["box_object"].get_point_indices_within_bounding_box(open3d.utility.Vector3dVector(aneurysm))))
                score=len(box_dict["box_object"].get_point_indices_within_bounding_box(open3d.utility.Vector3dVector(aneurysm)))/len(aneurysm)
                if score>scorebefore:
                    scorebefore=score
            total_score+=scorebefore    
    
    return total_score/(len(np.unique(labeled_aneurysm_mask))-1)

def bboxfit(boxobjects:List,labeled_aneurysm_mask:np.array):
    total_score=0
    shape= labeled_aneurysm_mask.shape
    for box_dict in boxobjects:
            scorebefore=0
            for label in range(1,int(np.unique(labeled_aneurysm_mask)[-1])+1):
                aneurysm = np.array(np.where(labeled_aneurysm_mask==label)).T
                score=calc_max_distance_to_box(box_dict["box_object"],aneurysm)
                if score>scorebefore:
                    scorebefore=score
            total_score+=scorebefore     
    return total_score#/len(np.unique(labeled_aneurysm_mask)-1)


def calc_max_distance_to_box(oriented_box,indices_aneurysm):
    max_boundaries=np.amax(np.linalg.inv(oriented_box.R).dot(np.array(oriented_box.get_box_points()).T).T,axis=0)
    min_boundaries=np.amin(np.linalg.inv(oriented_box.R).dot(np.array(oriented_box.get_box_points()).T).T,axis=0)
    max_coords_aneurysm =np.amax(np.linalg.inv(oriented_box.R).dot(indices_aneurysm.T).T,axis=0)
    min_coords_aneurysm =np.amin(np.linalg.inv(oriented_box.R).dot(indices_aneurysm.T).T,axis=0)
    max_distance = max(np.max(np.abs(np.subtract(max_boundaries,max_coords_aneurysm))),np.max(np.abs(np.subtract(min_boundaries,min_coords_aneurysm))))
    return max_distance


def f2_score(predicted_labeled_mask,groundtruth_labeled_mask):
    detected=0
    false_positive=0
    false_negative=0
    all_indices =list(np.array(np.where(predicted_labeled_mask>0)).T)
    all_indices = [tuple(x)for x in all_indices]
    for groundtruth_label in range(1,int(np.unique(groundtruth_labeled_mask)[-1])+1):
            true_indices = list(np.array(np.where(groundtruth_labeled_mask==groundtruth_label)).T)
            true_indices = [tuple(x)for x in true_indices]
            if any(x in all_indices for x in true_indices):
                detected+=1
            else:
                false_negative+=1

    all_indices_groundtruth =list(np.array(np.where(groundtruth_labeled_mask>0)).T)
    all_indices_groundtruth = [tuple(x)for x in all_indices_groundtruth]
    for label in range(1,int(np.unique(predicted_labeled_mask)[-1])+1):
        indices = list(np.array(np.where(predicted_labeled_mask==label)).T)
        indices = [tuple(x)for x in indices]
        if not any (x in all_indices_groundtruth for x in indices):
                false_positive+=1
    if detected ==0 and false_positive ==0 and false_negative ==0:
        f2=1.0
    elif detected ==0:
        f2=0.0
    else:
        P=detected/(detected+false_positive)
        R= detected/(detected+false_negative)
        f2= 5*P*R/(4*P+R)
    return f2



def calc_scores_task_1(labeled_masks,groundtruth_labeled_masks,boxes):
    scores_dict={"coverage_score":{"all_data":[]},"bbox_fit_score":{"all_data":[]},"f2_score":{"all_data":[]}}
    for predicted,groundtruth,box_dict in zip(labeled_masks,groundtruth_labeled_masks,boxes):
        scores_dict["coverage_score"]["all_data"].append(coverage(box_dict["candidates"],groundtruth))
        scores_dict["bbox_fit_score"]["all_data"].append(bboxfit(box_dict["candidates"],groundtruth))
        scores_dict["f2_score"]["all_data"].append(f2_score(predicted,groundtruth))
    for key in scores_dict.keys():
        scores_dict[key]["all_data"]=MinMaxScaler().fit_transform(np.array([scores_dict[key]["all_data"]]).T)
        scores_dict[key]["average"]=np.mean(scores_dict[key]["all_data"])
    return scores_dict

def calc_volume_bias(labeled_mask,groundtruth_labeled_mask):
    total_score=0
    for label in range(1,int(np.unique(labeled_mask)[-1])+1):
        indices = list(np.array(np.where(labeled_mask==label)).T)
        scorebefore=0
        for groundtruth_label in range(1,int(np.unique(groundtruth_labeled_mask)[-1])+1):
            true_indices = list(np.array(np.where(groundtruth_labeled_mask==groundtruth_label)).T)
            score= abs((len(true_indices)-len(indices)))
            if score<scorebefore:
                    scorebefore=score
            total_score+=scorebefore    
    return total_score/(len(np.unique(groundtruth_labeled_mask))-1)

def average_distance_score(labeled_mask,groundtruth_labeled_mask):
    total_score=0
    for label in range(1,int(np.unique(labeled_mask)[-1])+1):
        indices = list(np.array(np.where(labeled_mask==label)).T)
        scorebefore=0
        for groundtruth_label in range(1,int(np.unique(groundtruth_labeled_mask)[-1])+1):
            true_indices = list(np.array(np.where(groundtruth_labeled_mask==groundtruth_label)).T)
            score=calc_average_distance(indices,true_indices)
            if score<scorebefore:
                    scorebefore=score
            total_score+=scorebefore   
    return total_score/(len(np.unique(groundtruth_labeled_mask))-1)


def calc_average_distance(indices,true_indices):
    min_distances = np.amin(distance_matrix(indices,true_indices))/len(indices)
    min_distances_groundtruth=np.amin(distance_matrix(true_indices,indices))/len(true_indices)
    return min_distances+min_distances_groundtruth



def hausdorff_distance_score(labeled_mask,groundtruth_labeled_mask):
    total_score=0
    for label in range(1,int(np.unique(labeled_mask)[-1])+1):
        labeled= labeled_mask==label
        scorebefore=0
        for groundtruth_label in range(1,int(np.unique(groundtruth_labeled_mask)[-1])+1):
            true_labeled = groundtruth_labeled_mask==groundtruth_label
            score=hausdorff_distance(labeled,true_labeled)
            if score<scorebefore:
                    scorebefore=score
            total_score+=scorebefore   
    return total_score/(len(np.unique(groundtruth_labeled_mask))-1)

def calc_scores_task_2(mri_imgs,mri_imgs_ground_truth,labeled_masks,groundtruth_labeled_masks):
    scores_dict={"Jaccard":{"all_data":[]},"Haussdorf":{"all_data":[]},"Average_dist":{"all_data":[]},"Pearson_correlation":{"all_data":[]},"VolumeBias":{"all_data":[]}}
    for mask,ground_truth,labeled_mask,groundtruth_labeled in zip(mri_imgs,mri_imgs_ground_truth,labeled_masks,groundtruth_labeled_masks):
        scores_dict["Jaccard"]["all_data"].append(jaccard_score(ground_truth.flatten(),mask.flatten()))
        scores_dict["Haussdorf"]["all_data"].append(hausdorff_distance_score(mask,ground_truth))
        scores_dict["Pearson_correlation"]["all_data"].append(pearsonr(mask.flatten(),ground_truth.flatten())[0])
        scores_dict["VolumeBias"]["all_data"].append(calc_volume_bias(labeled_mask,groundtruth_labeled))
        scores_dict["Average_dist"]["all_data"].append(average_distance_score(labeled_mask,groundtruth_labeled))
    
    for key in scores_dict.keys():
        scores_dict[key]["all_data"]=MinMaxScaler().fit_transform(np.array([scores_dict[key]["all_data"]]).T)
        scores_dict[key]["average"]=np.mean(scores_dict[key]["all_data"])
        scores_dict["VolumeBias"]["stdev"]=np.std(scores_dict["VolumeBias"]["all_data"])
    return scores_dict

def calc_total_segmentation_score(scores_dict):
    total_score=0
    for key in scores_dict.keys():
        if key in ["Haussdorf","VolumeBias","Average_dist"]:
            if scores_dict[key]["average"]==0:
                total_score+=1
            else:
                total_score+=1/scores_dict[key]["average"]
        else:
            total_score+=scores_dict[key]["average"]

    total_score+=1/scores_dict["VolumeBias"]["stdev"]
    return total_score/6
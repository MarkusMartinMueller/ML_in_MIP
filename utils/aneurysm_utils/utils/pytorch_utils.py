import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset


def predict(model, images: list, cuda=False, apply_softmax=True):
    predictions = []
    for img in images:
        # Forward pass.
        image_tensor = torch.Tensor([img])  # convert numpy or list to tensor
        if cuda:
            image_tensor = image_tensor.cuda()
        X = torch.autograd.Variable(
            image_tensor[None], requires_grad=True
        )  # add dimension to simulate batch
        output = model(X)
        if apply_softmax:
            output = torch.nn.functional.softmax(output)

        # Backward pass.
        model.zero_grad()
        output_class = output.max(1)[1].data[0].item()
        output_probability = output.max(1)[0].data[0].item()
        predictions.append((output_class, output_probability))
    return predictions


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(
                    name,
                    ":",
                    "x".join(str(x) for x in list(param.size())),
                    "=",
                    num_param,
                )
            else:
                print(name, ":", num_param)
            total_param += num_param
    return total_param


class PPMIDataset(Dataset):
    """
    PyTorch dataset that consists of MRI images and labels.
    Args:
        mri_imgs (iterable of ndarries): The mri images.
        labels (iterable): The labels for the images.
        transform: Any transformations to apply to the images.
    """

    def __init__(
        self,
        mri_imgs,
        labels,
        transform=None,
        target_transform=None,
        z_factor=None,
        dtype=np.float32,
        num_classes=2,
    ):
        self.mri_imgs = np.copy(mri_imgs)
        self.labels =  np.copy(labels) #torch.LongTensor(labels)
        self.transform = transform
        self.target_transform = target_transform
        self.z_factor = z_factor
        self.dtype = dtype
        self.num_classes = num_classes

    def __len__(self):
        return len(self.mri_imgs)

    def __getitem__(self, idx):
        label = self.labels[idx].astype(np.float32)
        image = np.expand_dims(self.mri_imgs[idx], axis=0).astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

    def image_shape(self):
        """The shape of the MRI images."""
        return self.mri_imgs[0].shape

    def print_image(self, idx=None):
        """Function prints slices of a (random) selected mri"""
        if not idx:
            idx = random.randint(0, (self.mri_imgs.shape[0] - 1))
        dimension = len(self.mri_imgs[idx].shape)
        if dimension == 3:
            plt.subplot(1, 3, 1)
            plt.imshow(np.fliplr(self.mri_imgs[idx][:, :, 50]).T, cmap="gray")
            plt.subplot(1, 3, 2)
            plt.imshow(np.flip(self.mri_imgs[idx][:, 50, :]).T, cmap="gray")
            plt.subplot(1, 3, 3)
            plt.imshow(np.fliplr(self.mri_imgs[idx][50, :, :]).T, cmap="gray")
            plt.title(
                "Scans of id  " + str(idx) + "with label  " + str(self.labels[idx])
            )
            plt.show()


# -------------------------- Samplers ---------------------------------


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        counts = torch.bincount(dataset.tensors[1].squeeze())
        weights_ = [1.0 / 100, 1]

        weights = [weights_[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx].item()

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class ImbalancedDatasetSampler2(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx].item()

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples

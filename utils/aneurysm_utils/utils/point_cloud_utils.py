import numpy as np
from torch_geometric.data import Data
import torch


def segmented_image_to_graph(image: np.array, image_mask: np.array):
    """
    Converts orinal image to labeled graph

    Parameters
    ----------
    image
        image with one channel for intensity values
    mask
        array of same size, with 1 for aneuyrism
    Returns
    -------
    torch_geometric.data.Data
        contains the graph
    """
    labels = torch.tensor(np.ndarray.flatten(image_mask[image != 0]), dtype=torch.long)
    coordinates = torch.tensor(np.where(image != 0), dtype=torch.float)
    coordinates = torch.transpose(coordinates, 0, 1)
    intensity_values = torch.unsqueeze(torch.tensor(np.ndarray.flatten(image[image != 0]), dtype=torch.float), 1)
    return Data(x=intensity_values, pos=coordinates, y=labels)


def graph_to_image(graph: Data, shape: tuple = (256, 256, 220), attribute: str = "pred"):
    image = np.zeros(shape)
    # if pos normalizied
    #graph.pos[:, :2] = graph.pos[:, :2] * shape[0]
    #graph.pos[:, 2:] = graph.pos[:, 2:] * shape[2]
    for value, coords in zip(graph[attribute], graph.pos):
        coords = [round(num) for num in coords.tolist()]

        image[coords[0], coords[1], coords[2]] = value.item()
    return image


def extend_point_cloud(pred_classes, pred_scores, dataset, labels_test):
    pred_classes_rescaled = []
    prob_scores_rescaled = []
    for data, pred, prob, target in zip(dataset, pred_classes, pred_scores, labels_test):
        data.pred = pred
        pred_rescaled = graph_to_image(data, shape=target.shape, attribute="pred")
        data.prob = prob
        prob_rescaled = graph_to_image(data, shape=target.shape, attribute="prob")
        pred_classes_rescaled.append(pred_rescaled)
        prob_scores_rescaled.append(prob_rescaled)

    return pred_classes_rescaled, prob_scores_rescaled

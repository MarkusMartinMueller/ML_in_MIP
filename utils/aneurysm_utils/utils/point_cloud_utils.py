import numpy as np 
from torch_geometric.data import Data
import torch


    
    
def segmented_image_to_graph(image : np.array, image_mask: np.array):
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
    labels = torch.tensor(np.ndarray.flatten(image_mask[image!=0]),dtype = torch.float)
    coordinates =torch.tensor(np.where(image !=0),dtype=torch.float)
    intensity_values =torch.unsqueeze( torch.tensor(np.ndarray.flatten(image[image!=0]),dtype=torch.float),1)
    return Data(x=intensity_values,pos=coordinates,y=labels)


        
def graph_to_image(graph : Data,shape:tuple=(256,256,220)):
    image =np.zeros(shape)
    graph.pos[:,:2]=graph.pos[:,:2]*shape[0]
    graph.pos[:,2:]=graph.pos[:,2:]*shape[2]
    for value, coords in zip(graph.x,graph.pos):
        coords=[round(num) for num in coords.tolist()]

        image[coords[0],coords[1],coords[2]]= value.item()
    return image



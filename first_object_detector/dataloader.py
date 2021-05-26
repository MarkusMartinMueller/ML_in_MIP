import torch
from helpfunctions import creating_rectangles
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,type: str,num_objs,num_imgs=50000):
        X,y,bboxes,imgs =creating_rectangles(num_objs)
        i =int(0.8*num_imgs)
        if type== "test":
            self.labels =y[i:].astype(np.float32)
            self.data=X[i:].astype(np.float32)
            self.bboxes=bboxes[i:].astype(np.float32)
            self.imgs=imgs[i:].astype(np.float32)
            
        elif type =="train":
            self.labels=y[:i].astype(np.float32)
            self.data=X[:i].astype(np.float32)
            self.bboxes=bboxes[:i].astype(np.float32)
            self.imgs=imgs[:i].astype(np.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        X=self.data[index]
        y=self.labels[index]
        return X,y
    def get_imgs_bbox(self):
        
        return self.bboxes,self.imgs
    def getX(self):
        return self.data
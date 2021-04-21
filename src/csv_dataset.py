import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class ImageDataset(Dataset):
    def __init__(self, csv_path, mean=None, std=None):
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        self.mean = mean
        self.std = std
        # First column contains the canopy sizes
        self.size_arr = np.asarray(self.data_info.loc[:, 'area'])
        self.size_arr = np.array(self.size_arr, dtype=np.float32)
        self.size_arr = self.size_arr.reshape(-1,1)        
        
        # Second column is the labels/height
        self.label_arr = np.asarray(self.data_info.loc[:, 'height'])
        self.label_arr = np.array(self.label_arr, dtype=np.float32)
        self.label_arr = self.label_arr.reshape(-1,1)
        
        # Calculate len
        self.data_len = len(self.data_info.index)


    def __getitem__(self, index):

        size = self.size_arr[index]

        height = self.label_arr[index]

        # apply transform is one exists
        if self.mean is not None and self.std is not None:
            size = (size - self.mean) / self.std
            #height = (height - 156.2634) / 7.2301


        return (size, height)

    def setMean(self, mean):
        self.mean = mean
    
    def setSTD(self, std):
        self.std = std

    def __len__(self):
        return self.data_len


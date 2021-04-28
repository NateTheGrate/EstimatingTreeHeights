# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
import torch.nn as nn
import torch.nn.functional as F # Non-linearities package
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

class MnistCNNModel(nn.Module):
    def __init__(self):
        super(MnistCNNModel, self).__init__()
        #self.relu1 = nn.ReLU()
        #self.relu1 = nn.Sigmoid()
        self.fc1 = nn.Linear(1,1)
        #self.fc2 = nn.Linear(16,32)
        #self.fc3 = nn.Linear(32,16)
        #self.fc4 = nn.Linear(2,1)


    def forward(self, x):
        # Convolution 1
        #x = x.view(x.size(0), -1)
        out = self.fc1(x)
        #x = self.relu1(x)
        #x = self.fc2(x)
        #x = self.relu1(x)
        #x = self.fc3(x)
        #x = self.relu1(x)
        #x = self.fc4(x)
        #x = self.relu1(x)
        #print("after" ,x)
        return out

    def __len__(self):
        return self.data_len
    

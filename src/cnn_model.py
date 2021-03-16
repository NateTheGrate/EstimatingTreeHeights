# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
import torch.nn as nn
import torch.nn.functional as F # Non-linearities package
import torch

class MnistCNNModel(nn.Module):
    def __init__(self):
        super(MnistCNNModel, self).__init__()
        # Convolution 1
        #self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        #self.relu1 = nn.ReLU()
        self.relu1 = nn.Sigmoid()
        #self.maxpool1 = nn.MaxPool2d(kernel_size=4)
        #self.fc1 = nn.Linear(576, 10)
        self.fc1 = nn.Linear(1,2)
        #self.fc2 = nn.Linear(16,32)
        #self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(2,1)


    def forward(self, x):
        # Convolution 1
        x = x.view(x.size(0), -1)
        #print(x)
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.fc2(x)
        #x = self.relu1(x)
        #x = self.fc3(x)
        #x = self.relu1(x)
        x = self.fc4(x)
        x = self.relu1(x)
        #print("after" ,x)
        return x

    def __len__(self):
        return self.data_len
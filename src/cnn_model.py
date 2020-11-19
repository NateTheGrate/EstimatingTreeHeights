# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
import torch.nn as nn
import torch.nn.functional as F # Non-linearities package

class MnistCNNModel(nn.Module):
    def __init__(self):
        super(MnistCNNModel, self).__init__()
        # Convolution 1
        #self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        #self.relu1 = nn.ReLU()
        self.relu1 = nn.Sigmoid()
        #self.maxpool1 = nn.MaxPool2d(kernel_size=4)
        #self.fc1 = nn.Linear(576, 10)
        self.fc1 = nn.Linear(12 * 12 * 3, 6)
        self.fc2 = nn.Linear(6,2)


    def forward(self, x):
        # Convolution 1
        #out = self.cnn1(x)
        #out = self.relu1(out)
        #out = self.maxpool1(out)
        #out = out.view(out.size(0), -1)
        out = x.view(-1,12*12*3)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        #x = F.relu(self.fc1(x) )
        #x = F.relu(self.fc2(x))
        return out
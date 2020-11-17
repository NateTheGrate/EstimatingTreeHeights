# DISCLAIMER THIS IS NOT MY CODE. I FOLLOWED THIS TUTORIAL HERE: https://lelon.io/blog/pytorch-baby-steps

import torch # Tensor Package (for use on GPU)
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
import torch.optim as optim # Optimization package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models

import matplotlib.pyplot as plt # for plotting
import numpy as np


transform = transforms.Compose(
   [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.relu(self.fc1(x) )
        x = F.relu(self.fc2(x))
        return x

net = Net()#.cuda() not using nvidia gpu

NUMBER_OF_EPOCHS = 25
LEARNING_RATE = 1e-2
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

for epoch in range(NUMBER_OF_EPOCHS):
    train_loader_iter = iter(trainloader)
    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
        net.zero_grad()
        #inputs, labels = Variable(inputs.float().cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs.float()), Variable(labels)
        output = net(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
    if epoch % 5 is 0:
        print("Iteration: " + str(epoch + 1))

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
#outputs = net(Variable(images.cuda()))
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

correct = 0
total = 0
for data in testloader:
    images, labels = data
    #labels = labels.cuda()
    #outputs = net(Variable(images.cuda()))
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

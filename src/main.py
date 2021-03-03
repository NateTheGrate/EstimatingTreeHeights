import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision # for dealing with vision data

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# in-repo imports
from cnn_model import MnistCNNModel
import csv_dataset


TRAIN_CSV = "./data/csv/test.csv"
TEST_CSV = "./data/csv/test.csv"

if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Read image data from Csv 
    custom_mnist_from_csv_data = csv_dataset.ImageDataset(TRAIN_CSV, transform) 

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_data,
                                                    batch_size=1,
                                                    shuffle=True)

    # Read image data from Csv
    testset = csv_dataset.ImageDataset(TEST_CSV, transform) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)


    NUMBER_OF_EPOCHS = 50
    LEARNING_RATE = 0.1

    model = MnistCNNModel()
    model.train(True)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUMBER_OF_EPOCHS):
        trainloader_iter = iter(mn_dataset_loader)
        for batch_idx, (images, labels) in enumerate(trainloader_iter):
            images = Variable(images).float()
            labels = Variable(labels)
            labels = labels.unsqueeze(1).float()
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    total_avg_error = 0
    total_avg_error_p  = 0
    # test model
    for i, (images, labels) in enumerate(testloader):
        outputs = model(Variable(images).float())
        _,predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum()
        total_avg_error += (predicted - labels)
        total_avg_error_p += total_avg_error/labels


    total_avg_error = total_avg_error/total
    total_avg_error_p = total_avg_error_p/total

    print(total_avg_error, total_avg_error_p)

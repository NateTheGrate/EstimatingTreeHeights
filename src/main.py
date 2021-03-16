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


TRAIN_CSV = "./data/csv/canopies.csv"
TEST_CSV = "./data/csv/canopies.csv"

if __name__ == "__main__":


    # Read image data from Csv 
    custom_mnist_from_csv_data = csv_dataset.ImageDataset(TRAIN_CSV) 

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_data,
                                                    batch_size=1,
                                                    shuffle=True)

    # Read image data from Csv
    testset = csv_dataset.ImageDataset(TEST_CSV) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)


    NUMBER_OF_EPOCHS = 50000
    LEARNING_RATE = 0.0005

    model = MnistCNNModel()
    model.train()
    criterion = nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUMBER_OF_EPOCHS):
        trainloader_iter = iter(mn_dataset_loader)
        for batch_idx, (images, labels) in enumerate(trainloader_iter):
            images = Variable(images)
            labels = Variable(labels)
            #labels = labels.unsqueeze(1).float()
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
        temp = 0
        labels = Variable(labels)
        outputs = model(Variable(images))
        predicted = outputs * 10 #torch.max(outputs.data, 1)
        labels *= 10
        total += labels.size(0)
        temp += ((predicted - labels.item()))
        total_avg_error_p += total_avg_error/temp
        total_avg_error += abs(temp)
        print("Height: ", labels.item(), " Predicted height: ", predicted.item())


    total_avg_error = total_avg_error/total
    total_avg_error_p = total_avg_error_p/total

    print("Average error:", round(total_avg_error.item(),2), "feet")
    print("Average percent error:", round(total_avg_error_p.item(),2), "%")

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


TRAIN_CSV = "./data/csv/testcsv.csv"
TEST_CSV = "./data/csv/testcsvtester.csv"

if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Read image data from Csv 
    custom_mnist_from_csv_data = csv_dataset.ImageDataset(TRAIN_CSV, transform) 

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_data,
                                                    batch_size=3,
                                                    shuffle=False)

    # Read image data from Csv
    testset = csv_dataset.ImageDataset(TEST_CSV, transform) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)


    NUMBER_OF_EPOCHS = 25
    LEARNING_RATE = 1e-1

    model = MnistCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUMBER_OF_EPOCHS):
        trainloader_iter = iter(mn_dataset_loader)
        for batch_idx, (images, labels) in enumerate(trainloader_iter):
            images = Variable(images)
            labels = Variable(labels)
            
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
            

    correct = 0
    total = 0
    # test model
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print(predicted)
    

    f, axarr = plt.subplots(nrows=1,ncols=4, figsize=(12,4))
    
    for i, (images, labels) in enumerate(testloader):
        # get data from model about image
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        
        if(i >= 3):
            prd = ""
            if(predicted == torch.tensor([1])):
                prd = "True"
            else:
                prd = "false"

            lbl = ""
            if(labels == torch.tensor([1])):
                lbl = "true"
            else:
                lbl = "false"

            img = (torchvision.utils.make_grid(images))
            img = img / 2 + 0.5
            npimg = img.numpy()

            plt.sca(axarr[i-3])
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

            
            plt.title("Predicted: " + prd + "\nActual: " + lbl)
            plt.axis('off')

        if(i >= 6):
            break

    plt.savefig("./data/figures/graph.png")

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from os import sys

# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# in-repo imports
from cnn_model import MnistCNNModel
import csv_dataset
import dataentry.image_processing as ip
import knn

TRAIN_CSV = "./data/csv/canopiesFromHighestHit.csv"
TEST_CSV = "./data/csv/canopiesFromHighestHit.csv"
COLOR_IMAGE = './data/images/training/highest_hit.png'


# returns weight, bias, mean, std
def train(train_csv, num_of_epochs=10000, learning_rate=0.0001):
    # Read image data from Csv 
    train_set = csv_dataset.ImageDataset(train_csv) 

    # use data loader to find mean and std
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=len(train_set),
                                                num_workers=1)

    data = next(iter(mn_dataset_loader))
    mean = data[0].mean().item()
    std = data[0].std().item()

    train_set.setMean(mean)
    train_set.setSTD(std)

    # now reset data loader for training
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=1,
                                                shuffle=True)



    # uncomment for multicore
    #torch.set_num_threads(4)

    model = MnistCNNModel()
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in trange(num_of_epochs, desc='Training model', unit="carrots"):
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

    # returns weight, bias
    return model.fc1.weight.data.numpy()[0][0], model.fc1.bias.data.numpy()[0], mean, std


def evaluate(test_csv, weight, bias, mean, std):


    # Read image data from Csv
    testset = csv_dataset.ImageDataset(test_csv) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)

      # setup variables for testing
    x = np.linspace(0, 6000, 3000)
    
    # line model produces adjusting for normalization
    line = weight * (x - mean)/std + bias
    xs = []
    ys = []

    total_avg_error = 0
    for i, (images, labels) in enumerate(testloader):
        labels = Variable(labels)
        images = Variable(images)
        # calculate output based on linear line generated
        output = weight * (images - mean)/std + bias
        # scatter plot data
        xs.append(images)
        ys.append(labels)

        total_avg_error += abs(output - labels)
        print("Height: ", labels.item(), " Predicted height: ", output.item())

    plt.scatter(xs,ys)
    plt.plot(x,line, '-r')
    plt.savefig('./data/figures/output.png')
    
    total_avg_error = total_avg_error/len(testloader)
    print("Average absolute error:", round(total_avg_error.item(),2), "feet")
    #print("Average percent error:", round(total_avg_error_p.item(),2), "%")


def generate_csv(color_image, demo):
    # if demo is true it will fill out a csv for testing with heights as -1
    if demo:
        ip.process_image(COLOR_IMAGE, TEST_CSV, demo)
    else:
        ip.process_image(COLOR_IMAGE, TRAIN_CSV, demo)

if __name__ == "__main__":

    n = len(sys.argv)
    demo = False
    knn = False
    if n > 1:
        demo = sys.argv[1]
    elif n > 2:
        TRAIN_CSV = sys.argv[2]
    elif n > 3:
        TEST_CSV = sys.argv[3]
    elif n > 4:
        COLOR_IMAGE = sys.argv[4]
    elif n > 5:
        knn = sys.argv[5]
    
    
    print("generating csv...")
    #generate_csv(COLOR_IMAGE, demo)
    print("csv generated")

    print("training...")
    if not knn:
        weight, bias, mean, std = train(TRAIN_CSV, 100)
        evaluate(TRAIN_CSV, weight, bias, mean, std)
    else:
        knn.train_data = TRAIN_CSV
        knn.test_data = TEST_CSV
        knn.cross_validation(4)
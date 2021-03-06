import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from os import sys
import pandas as pd
# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# in-repo imports
from cnn_model import MnistCNNModel
import csv_dataset
import dataentry.image_processing as ip
import knn

TRAIN_CSV = "./data/csv/canopiesFromHighestHit.csv"
TEST_CSV = "./data/csv/canopiesFromHighestHit.csv"
COLOR_IMAGE = './data/images/training/color.png'


# returns weight, bias, mean, std
def train(train_csv, num_of_epochs=10000, learning_rate=0.0001):
    # Read image data from Csv 
    train_set = csv_dataset.ImageDataset(train_csv) 

    # use data loader to find mean and std
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=len(train_set),
                                                num_workers=1)

    # normalize data
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
    criterion = nn.L1Loss()
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
    df = pd.read_csv(test_csv)

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
    losses = []
    for i, (images, labels) in enumerate(testloader):
        labels = Variable(labels)
        images = Variable(images)
        # calculate output based on linear line generated
        output = weight * (images - mean)/std + bias
        # scatter plot data
        xs.append(images)
        ys.append(labels)
        losses.append(round(output.item() - labels.item(), 2))
        total_avg_error += abs(output - labels)
        print("Height: ", labels.item(), " Predicted height: ", output.item())

    plt.scatter(xs,ys)
    plt.plot(x,line, '-r')
    plt.savefig('./data/figures/output.png')
    
    total_avg_error = total_avg_error/len(testloader)
    print("Average absolute error:", round(total_avg_error.item(),2), "feet")
    df['losses'] = losses
    ip.add_height_markers_df(COLOR_IMAGE, df)
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
    is_knn = True
    if n > 1:
        demo = sys.argv[1]
    elif n > 2:
        TRAIN_CSV = sys.argv[2]
    elif n > 3:
        TEST_CSV = sys.argv[3]
    elif n > 4:
        COLOR_IMAGE = sys.argv[4]
    elif n > 5:
        is_knn = sys.argv[5]
    
    
    print("generating csv...")
    #generate_csv(COLOR_IMAGE, demo)
    print("csv generated")

    print("training...")
    if not is_knn:
        weight, bias, mean, std = train(TRAIN_CSV, 100)
        evaluate(TRAIN_CSV, weight, bias, mean, std)
        
        
    else:
        knn.train_data = TRAIN_CSV
        knn.test_data = TEST_CSV
        losses = knn.evaluate(TEST_CSV, 4)

        df = pd.read_csv(TRAIN_CSV)
        df['losses'] = losses

        ip.add_height_markers_df(COLOR_IMAGE, df)


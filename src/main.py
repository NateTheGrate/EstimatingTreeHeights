import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision # for dealing with vision data
from tqdm import trange
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# in-repo imports
from cnn_model import MnistCNNModel
import csv_dataset


TRAIN_CSV = "./data/csv/canopiesFromHighestHit.csv"
TEST_CSV = "./data/csv/canopiesFromHighestHit.csv"

if __name__ == "__main__":


    # Read image data from Csv 
    train_set = csv_dataset.ImageDataset(TRAIN_CSV) 

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

    # Read image data from Csv
    testset = csv_dataset.ImageDataset(TEST_CSV) 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)


    NUMBER_OF_EPOCHS = 100
    LEARNING_RATE = 0.0001

    torch.set_num_threads(4)
    print(torch.get_num_threads())
    model = MnistCNNModel()
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for i in trange(NUMBER_OF_EPOCHS, desc='Training model', unit="carrots"):
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
    x = np.linspace(0, 6000, 3000)
    line = model.fc1.weight.data.numpy()[0][0] * (x - mean)/std + model.fc1.bias.data.numpy()[0]
    xs = []
    ys = []
    # test model
    for i, (images, labels) in enumerate(testloader):
        temp = 0
        labels = Variable(labels)
        outputs = model(Variable(images))
        xs.append(images)
        ys.append(labels)
        predicted = outputs #torch.max(outputs.data, 1)
        total += labels.size(0)
        total_avg_error += abs(temp)
        print("Height: ", labels.item(), " Predicted height: ", predicted.item())

    plt.scatter(xs,ys)
    plt.plot(x,line, '-r')
    plt.savefig('./data/figures/output.png')
    total_avg_error = total_avg_error/total

    print("Average error:", round(total_avg_error,2), "feet")
    #print("Average percent error:", round(total_avg_error_p.item(),2), "%")

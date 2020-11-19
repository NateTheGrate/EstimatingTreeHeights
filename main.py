import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision # for dealing with vision data

import numpy as np
from PIL import Image


# in-repo imports
from cnn_model import MnistCNNModel
import loaddata


if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])

    # Dataset variant 1:
    # Read image data from Csv - torch transformations are taken as argument
    custom_mnist_from_csv_data = loaddata.ImageDataset("./testcsv.csv") #\
        #CustomDatasetFromCsvData('./testcsv.csv',
         #                        12, 12,
          #                       transformations)


    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_data,
                                                    batch_size=1,
                                                    shuffle=True)

        # Read image data from Csv - torch transformations are taken as argument
    testset = loaddata.ImageDataset("./testcsvtester.csv") 
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                                    batch_size=1,
                                                    shuffle=False)


    NUMBER_OF_EPOCHS = 25
    LEARNING_RATE = 1e-2

    model = MnistCNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUMBER_OF_EPOCHS):
        for i, (images, labels) in enumerate(mn_dataset_loader):
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
            break
    
    
    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        total_width = 48
        max_height = 12
        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_im.save('')
        #plt.savefig("graph.png")

    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        

        
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

# credit to this git repo: https://github.com/utkuozbulak/pytorch-custom-dataset-examples
class ImageDataset(Dataset):
    def __init__(self, csv_path, transform):
        # Transforms
        self.to_tensor = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])

        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image as rgb
        img_as_img = Image.open(single_image_name).convert("RGB")
         # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


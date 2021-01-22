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

        # First column contains the canopy sizes
        self.size_arr = np.asarray(self.data_info.iloc[:, 1])
        
        # Second column is the labels/height
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])

        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        # Open image as rgb
        #img_as_img = Image.open(single_image_name).convert("RGB")
         # Transform image to tensor
        size = self.size_arr[index]

        # Get label(class) of the image based on the cropped pandas column
        height = self.label_arr[index]

        return (size, height)

    def __len__(self):
        return self.data_len


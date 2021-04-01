import pandas as pd
import numpy as np
from PIL import Image
import PIL

DTM_MIN, DTM_MAX = float(3361.916504), float(3621.498291)
DSM_MIN, DSM_MAX = float(3362.287598), float(3627.270752)

DSM_IMG = './data/images/training/dsm2010.png'
DTM_IMG = './data/images/training/dtm2010.png'
HEIGHT_CSV = './data/csv/canopies.csv'

PIL.Image.MAX_IMAGE_PIXELS = 152463601

im = Image.open(DSM_IMG).convert('L')
dsmarray = np.array(im)

im2 = Image.open(DTM_IMG).convert('L')
dtmarray = np.array(im2)

# dsm height is absolute surface height (tree height from sea level) (in feet)
def pixelValToDSMHeight(pixelValue):
    return pixelValue + ((DSM_MAX - DSM_MIN) / 255.0) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue + ((DTM_MAX - DTM_MIN) / 255.0) + DTM_MIN

def appendHeights (dsm, dtm, csv):
    # read csv data
    data = csv
    for i in range(0, data.shape[0]):
        # get coords from csv (first 2 columns)
        x,y = np.asarray(data.iloc[i,:2])
        x = int(x)
        y = int(y)
        height =  pixelValToDSMHeight(dsm[y][x]) - pixelValToDTMHeight(dtm[y][x])
        data.loc[i,'height'] = height
        

    data.to_csv(HEIGHT_CSV, index=False, float_format='%.16g')
    print(data)


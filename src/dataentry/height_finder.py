import pandas as pd
import numpy as np
from PIL import Image
import PIL

DTM_MIN, DTM_MAX = float(2651.41), float(3413.2)
DSM_MIN, DSM_MAX = float(2651.71), float(3445.11)

DSM_IMG = './data/images/dsm2010.png'
DTM_IMG = './data/images/dtm2010.png'
HEIGHT_CSV = './data/csv/test.csv'

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

# read csv data
data = pd.read_csv(HEIGHT_CSV)
for i in range(0, data.shape[0]):
    # get coords from csv (first 2 columns)
    x,y = np.asarray(data.iloc[i,:2])
    x = int(x)
    y = int(y)
    height =  pixelValToDSMHeight(dsmarray[y][x]) - pixelValToDTMHeight(dtmarray[y][x])
    data.loc[i,'height'] = height * 0.1
    

data.to_csv(HEIGHT_CSV, index=False, float_format='%.16g')
print(data)


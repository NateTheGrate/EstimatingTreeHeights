import pandas as pd
import numpy as np
from PIL import Image

DTM_MIN, DTM_MAX = 2651, 3413.2
DSM_MIN, DSM_MAX = 2651.71, 3445.11

DSM_IMG = './data/images/dsm2010.png'
DTM_IMG = './data/images/dtm2010.png'
HEIGHT_CSV = './data/csv/test.csv'


im = Image.open(DSM_IMG).convert('L')
dsmarray = np.array(im)

im2 = Image.open(DTM_IMG).convert('L')
dtmarray = np.array(im2)

# dsm height is absolute surface height (tree height from sea level) (in feet)
def pixelValToDSMHeight(pixelValue):
    return pixelValue + ((DSM_MAX - DSM_MIN) / 255) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue + ((DTM_MAX - DTM_MIN) / 255) + DTM_MIN

# read csv data
data = pd.read_csv(HEIGHT_CSV)

for i in range(0, data.shape[0]):
    # get coords from csv (first 2 columns)
    x,y = np.asarray(data.iloc[i,:2])

    height =  pixelValToDSMHeight(dsmarray[y][x]) - pixelValToDTMHeight(dtmarray[y][x])

    data.iloc[i][3] = height

data.to_csv(HEIGHT_CSV, index=False)


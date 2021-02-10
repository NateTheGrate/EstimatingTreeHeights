# (x,y), canopy_size, tree_height (dogami)

import pandas as pd
import numpy as np
from PIL import Image

DTM_MIN, DTM_MAX = 2651, 3413.2
DSM_MIN, DSM_MAX = 2651.71, 3445.11


im = Image.open('./data/images/dsm2010.png').convert('L')
dsmarray = np.array(im)

im2 = Image.open('./data/images/dtm2010.png').convert('L')
dtmarray = np.array(im2)

# dsm height is absolute surface height (tree height) (in feet)
def pixelValToDSMHeight(pixelValue):
    return pixelValue + ((DSM_MAX - DSM_MIN) / 255) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue + ((DTM_MAX - DTM_MIN) / 255) + DTM_MIN


data = pd.read_csv('./data/csv/test.csv')
x,y = np.asarray(data.iloc[0,:2])
height =  pixelValToDSMHeight(dsmarray[y][x]) - pixelValToDTMHeight(dtmarray[y][x])
print(height)

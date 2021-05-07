import pandas as pd
import numpy as np
from PIL import Image
import PIL

DTM_MIN, DTM_MAX = float(2664.69), float(3413.08)
DSM_MIN, DSM_MAX = float(2665.12), float(3430.35)

DSM_IMG = './data/images/training/highest_hit.png'
DTM_IMG = './data/images/training/bare_earth.png'
HEIGHT_CSV = './data/csv/canopiesFromHighestHit.csv'

PIL.Image.MAX_IMAGE_PIXELS = 152463601

im = Image.open(DSM_IMG).convert('L')
dsmarray = np.array(im)

im2 = Image.open(DTM_IMG).convert('L')
dtmarray = np.array(im2)


# dsm height is absolute surface height (tree height from sea level) (in feet)
def pixelValToDSMHeight(pixelValue):
    return pixelValue * ((DSM_MAX - DSM_MIN) / 255.0) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue * ((DTM_MAX - DTM_MIN) / 255.0) + DTM_MIN

def appendHeights (dsm, dtm, csv):
    # read csv data
    data = pd.read_csv(csv)
    
    biggest = 0
    lowest = 255
    numrows = data.shape[0]
    for i in range(0, numrows):
        # get coords from csv (first 2 columns)
        x,y = np.asarray(data.iloc[i,1:3])
        x = int(x)
        y = int(y)

        height =  pixelValToDSMHeight(dsm[y][x]) - pixelValToDTMHeight(dtm[y][x])
        data.loc[i,'height'] = height 

        #if(height < 10 or height > 35):
        #    data.drop(labels=i, axis=0)
        #    numrows -= 1
        #    i = 0

    data.to_csv(HEIGHT_CSV, index=False, float_format='%.16g')
    print(data)

appendHeights(dsmarray, dtmarray, HEIGHT_CSV)
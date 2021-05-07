from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from PIL import Image
import PIL


DTM_MIN, DTM_MAX = float(3361.916504), float(3621.498291)
DSM_MIN, DSM_MAX = float(3362.287598), float(3627.270752)

DSM_IMG = './data/images/training/highest_hit.png'
DTM_IMG = './data/images/training/bare_earth.png'

PIL.Image.MAX_IMAGE_PIXELS = 152463601

# dsm height is absolute surface height (tree height from sea level) (in feet)
def pixelValToDSMHeight(pixelValue):
    return pixelValue * ((DSM_MAX - DSM_MIN) / 255.0) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue * ((DTM_MAX - DTM_MIN) / 255.0) + DTM_MIN

# at (x,y), subtract dsm from dtm (tree height from ground height)
def findHeights(dsm, dtm, x, y):
    height =  pixelValToDSMHeight(dsm[y][x]) - pixelValToDTMHeight(dtm[y][x])
    return height

def add_height_markers(image_path, csv):
    
    img = cv2.imread(image_path)
    data = pd.read_csv(csv)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0,0,255)
    lineType = 2

    for i in range(0, data.shape[0]):
        # get coords from csv (first 2 columns)
        x,y = np.asarray(data.iloc[i,1:3])
        x = int(x)
        y = int(y)
        height = data.loc[i,'height']
        img = cv2.putText(img, str(height), (x,y), font, fontScale, fontColor, lineType)
    

    #Save image
    cv2.imwrite("./data/figures/labeled_trees.png", img)



def add_height_markers_df(image_path, df):
    
    img = cv2.imread(image_path)
    data = df

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0,0,255)
    lineType = 2

    for i in range(0, data.shape[0]):
        # get coords from csv (first 2 columns)
        x,y = np.asarray(data.iloc[i,1:3])
        x = int(x)
        y = int(y)
        losses = data.loc[i,'losses']
        img = cv2.putText(img, str(losses), (x,y), font, fontScale, fontColor, lineType)
    

    #Save image
    cv2.imwrite("./data/figures/labeled_trees.png", img)



def process_image(image_path, csv_path, new_data):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgy = len(img)
    imgx = len(img[0])
    print("Reading image with width " + str(imgx) + " and height " + str(imgy) + ".")
    time1 = datetime.datetime.now().replace(microsecond=0)

    # Display in-process images
    debug = True
    # Average tree areas in pixels, the program checks that all contours are not too far away from this
    AVG_TREE_AREA = 2500
    # Grayscale colors used to display different parts of the image during processing
    COLOR_BLACK = 1
    COLOR_GRAY = 200
    # Thresholds used to control the sensitivity of the edge finding, not currently very algorithm-important
    CANNY_THRESH_LOW = 30
    CANNY_THRESH_HIGH = 40

    # Highlight harder to find tree edges by comparing neighboring color values in the height map
    # This currently doesn't help much with contouring, but helps with visualization and error estimation
    for i in range(imgy-2):
            for j in range(imgx-2):
                    if float(img[i+1][j+1][0]) - float(img[i][j][0]) > float(6):
                            img[i][j] = 0

    # Find spots of the image that are tall enough in the height map
    # Smooth out background into one gray color
    # 140 seems to be the sweet spot pixel value for tree canopy
    for i in range(imgy-1):
            for j in range(imgx-1):
                    if gray[i][j] > 140:
                            gray[i][j] = COLOR_BLACK
                    else:
                            gray[i][j] = COLOR_GRAY

    # Generate edges and contours
    edged = cv2.Canny(gray, CANNY_THRESH_LOW, CANNY_THRESH_HIGH)
    ret, thresh = cv2.threshold(edged, 127, 255, 0)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    validatedContours = []

    # Sanity check to remove large tree blobs or bushes
    for i in contours:
        a = cv2.contourArea(i)
        if AVG_TREE_AREA / 2 < a < AVG_TREE_AREA * 2:
            validatedContours = validatedContours + [i]


    i = cv2.drawContours(img,validatedContours, -1, (123,255,123), 3)

    canopies = {'x': [], 'y': [], 'area': [], 'height': []}

    # now time to get dsm and dtm bitmapas
    dsm_im = Image.open(DSM_IMG).convert('L')
    dsmarray = np.array(dsm_im)

    dtm_im = Image.open(DTM_IMG).convert('L')
    dtmarray = np.array(dtm_im)

    for c in validatedContours:
        m = cv2.moments(c)
        # Center
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

        height = -1
        # don't try to find heights in dsm, dtm in testing data
        if not new_data:
            height = abs(findHeights(dsmarray, dtmarray, cx, cy))

        # Some trees were showing up with no height
        if img[cy][cx][0] != 0:
            if height < 10 or height > 35:
                continue
            canopies['x'].append(cx)
            canopies['y'].append(cy)
            canopies['area'].append(cv2.contourArea(c)/100)
            canopies['height'].append(height)


    csv = pd.DataFrame.from_dict(canopies)
    if not new_data:
        csv.to_csv(csv_path, index=False, float_format='%.16g')
    else:
        csv.to_csv(csv_path, index=True, float_format='%.16g')

    time2 = datetime.datetime.now().replace(microsecond=0)
    print("Process finished in " + str(time2-time1))

    if debug:
        cv2.imshow('gray', gray)
        cv2.imshow('edged', edged)
        cv2.imshow('image', i)
        cv2.waitKey(0)
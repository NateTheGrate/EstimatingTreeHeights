from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv
import numpy as np
import pandas as pd
from skimage.segmentation import watershed
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray



def rgb_hsv(r, g, b):
    red = r/255
    green = r/255
    blue = r/255
    return rgb_to_hsv([red, green, blue])


# Read image
img = cv2.imread("./data/images/colorimage1.png")
og = img


lowShadow = np.array([0,0,0])
highShadow = np.array([180,255,5])

#cv2.imshow("cam", img)

mask = cv2.inRange(img, lowShadow, highShadow)
mask = cv2.bitwise_not(mask)

#cv2.imshow("mask", mask)
#cv2.imshow("image", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = watershed(img,markers)
#markers = cv2.bitwise_and(markers,markers, mask=mask)

mcopy = markers.copy()

cv2.imwrite("./temp.png",markers)

contours = cv2.findContours(markers, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(markers,contours, 0, (123,255,123), 3)

canopies = {}
canopies['x'] = []
canopies['y'] = []
canopies['area'] = []
canopies['height'] = []

for c in contours:
    m = cv2.moments(c)
    # Center
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])

    canopies['x'].append(cx)
    canopies['y'].append(cy)
    canopies['area'].append(cv2.contourArea(c)/100)
    canopies['height'].append(-1)


#print(canopies)
#print(len(contours))
#print(len(canopies))
#print(len(canopies['area']))

csv = pd.DataFrame.from_dict(canopies)
csv.to_csv('./data/csv/canopies.csv', index=False)


#fig, axs = plt.subplots (2,2)
#axs[0,0].imshow(mcopy)
#axs[0,1].imshow(canopyMask, cmap="gray")
#axs[0,1].imshow(markers)
#axs[1,0].imshow(og)
#plt.show()
from PIL import Image
import os

CROPSIZE = 12 # cropsize = x produces 12x12 cropped images
FULL_IMAGE_PATH = "./data/images/full_size_images/test.tif"
TRAINING_IMAGE_PATH = "./data/images/cropped_train/" 
TESTING_IMAGE_PATH = "./data/images/cropped_test"

im = Image.open(FULL_IMAGE_PATH)

width, height = 240,240 #im.size

k = 0
# rows
for i in range(0, width - CROPSIZE, CROPSIZE):
    # columns
    for j in range(20, height - CROPSIZE, CROPSIZE):
        # crop image with square of (i,j), (i+cropsize, j+cropsize)
        cropped = im.crop((i, j, i+CROPSIZE, j+CROPSIZE))

        #just doing images in order for now
        cropped.save(TRAINING_IMAGE_PATH + "/croptest_" + str(k)+ "_.tif")
        cropped.save(TESTING_IMAGE_PATH + "/croptest_" + str(k)+ "_.tif")
        k += 1

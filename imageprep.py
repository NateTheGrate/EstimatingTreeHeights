from PIL import Image

CROPSIZE = 12 # cropsize = x produces 12x12 cropped images

im = Image.open("./images/full_images/test.tif")

width, height = 240,240 #im.size
k = 0
# rows
for i in range(0, width - CROPSIZE, CROPSIZE):
    # columns
    for j in range(0, height - CROPSIZE, CROPSIZE):
        # crop image with square of (i,j), (i+cropsize, j+cropsize)
        cropped = im.crop((i, j, i+CROPSIZE, j+CROPSIZE))
        # cropped images have format (y, x)
        #cropped.save("./images/cropped_images/croptest_" + str(i) + "," + str(j) +"_.tif")
       
        #just doing images in order for now
        cropped.save("./images/cropped_images/croptest_" + str(k)+ "_.tif")
        k += 1

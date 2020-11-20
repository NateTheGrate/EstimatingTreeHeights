import glob
from cv2 import cv2
import csv

TRAINING_IMAGE_PATH = "./data/images/cropped_train/" 
TESTING_IMAGE_PATH = "./data/images/cropped_test"
TRAIN_CSV = "./data/csv/testcsv.csv"
TEST_CSV = "./data/csv/testcsvtester.csv"


images = glob.glob("./data/images/cropped_train/*.tif")
data = []
i = 0
for image in images:
    with open(image, 'rb') as file:
        img = cv2.imread(image)
        cv2.imshow('name', img)
        keypress = cv2.waitKey(0)

        entry = []
        entry.append(TRAINING_IMAGE_PATH + "/croptest_" + str(i)+ "_.tif")
        if keypress == ord('1'):
            # do stuff
            entry.append('1')
        elif keypress == ord('q'):
            break
        else:
            # do other stuff
            entry.append('0')

        data.append(entry)
        print(data)
        cv2.destroyAllWindows()
        i+=1

with open(TRAIN_CSV, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)
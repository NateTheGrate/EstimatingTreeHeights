import numpy as np
import sys


# get arrays from CSVs
train_data = np.genfromtxt('./data/csv/canopiesFromHighestHit.csv', delimiter=',', skip_header=1)
#test_data = np.genfromtxt('./test_pub.csv', delimiter=',', skip_header=1)


DTM_MIN, DTM_MAX = float(2664.69), float(3413.08)
DSM_MIN, DSM_MAX = float(2665.12), float(3430.35)

# dsm height is absolute surface height (tree height from sea level) (in feet)

def pixelValToDSMHeight(pixelValue):
    # pixel value = [0, 255]
    return pixelValue * ((DSM_MAX - DSM_MIN) / 255.0) + DSM_MIN

# dtm height is absolute ground height (in feet)
def pixelValToDTMHeight(pixelValue):
    return pixelValue * ((DTM_MAX - DTM_MIN) / 255.0) + DTM_MIN

def get_height(pixelValue):
    height = pixelValToDSMHeight(pixelValue) - pixelValToDTMHeight(pixelValue)
    return height


# returns k nearest neighbors of a data point not including itself
def knn(data_point, training_set, k, test=False):
    
    A = training_set[:, 3:-2]
    z = data_point[3:-2]

    # compute L2 distances from z to every other point in the training set
    Y = np.linalg.norm(A - z, axis=1, ord=1)

    # find indicies of nearest neighbors
    result = []
    for i in np.argsort(Y)[0:k]:
        # return k nearest neighbors' ids
        result.append(training_set[i, 0])

    return result


def find_row_value(id):
    for row in train_data:
        if row[0] == id:
            # return height
            return row[-2]
    return None

def count(nns, k):
    # average neighbor's heights
    total = 0
    for id in nns:
        total += find_row_value(id)

    return total/len(nns)

# return whether or not model thinks a data_point earns above
def predict(data_point, training_set, k, t=False):
    tally = count(knn(data_point, training_set, k, test=t), k)
    # if at least half of the neighbors have an income >50k
    # predict 1
    return tally


# 4-fold cross validation
# returns array size 4 of absolute error for each fold
def cross_validation(k):
    # split array into 4 parts
    a, b, c, d = np.array_split(train_data, 4)
    results = []

    # test a
    training_set = np.concatenate((b, c, d))
    total = 0
    for row in a:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - find_row_value(id))

    results.append(get_height(total/len(a)))

    # test b
    training_set = np.concatenate((a, c, d))
    total = 0
    for row in b:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - find_row_value(id))

    results.append(get_height(total/len(b)))

    # test c
    training_set = np.concatenate((a, b, d))
    total = 0
    for row in c:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - find_row_value(id))
    results.append(get_height(total/len(c)))

    # test d
    training_set = np.concatenate((a, b, c))
    total = 0
    for row in d:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - find_row_value(id))

    results.append(get_height(total/len(d)))

    return results

# accuracy on the whole training set against itself
# returns the losses for each entry in the training set in order and the absolute error
def training_accuracy(k):

    total = 0
    losses = []
    # record absolute error across the whole training set
    for i in range(0, len(train_data)):
        losses.append(predict(train_data[i], train_data, k) - train_data[i, -2])
        total += abs(predict(train_data[i], train_data, k) - train_data[i, -2])

    return losses, get_height(total / len(train_data))


# wrapper function for training accuracy and cross validation
# returns abolute error losses, training accuracy, and crossvalidation accuracies
def training_stats(train_csv, K):

    global train_data 
    train_data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

    # get training accuracy
    losses,trn_acc = training_accuracy(K)
    # get cross validation accuracies
    cv_accs = np.array(cross_validation(K))

    return losses, trn_acc, cv_accs



# returns heights for each entry in the test data in order
def evaluate(traincsv, testcsv, K):
    global train_data 
    train_data = np.genfromtxt(traincsv, delimiter=',', skip_header=1)
    test_data = np.genfromtxt(testcsv, delimiter=',', skip_header=1)
    
    heights = []
    for row in test_data:
        heights.append(get_height(predict(row, train_data, K)))

    return heights





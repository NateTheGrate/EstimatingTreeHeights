import numpy as np

# get arrays from CSVs
train_data = np.genfromtxt('./canopiesFromHighestHit.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('./test_pub.csv', delimiter=',', skip_header=1)


DTM_MIN, DTM_MAX = float(3361.916504), float(3621.498291)
DSM_MIN, DSM_MAX = float(3362.287598), float(3627.270752)

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

#7
# returns k nearest neighbors of a data point not including itself
def knn(data_point, training_set, k, test=False):
    A = training_set[:, 3:-1]
    z = data_point[3:-1]

    # adjust if testing data
    #if(test):
        #z = data_point[1:]

    # compute L2 distances from z to every other point in the training set
    Y = np.linalg.norm(A - z, axis=1)

    # find indicies of nearest neighbors
    result = []
    for i in np.argsort(Y)[0:k]:
        # return k nearest neighbors' ids
        result.append(training_set[i, 0])

    return result


# returns 1 if income >50k, 0 otherwise
def count(nns, k):
    # count how many nearest neighbors have an income greater than 50k
    total = 0
    for id in nns:
        total += train_data[int(id), -1]

    return total/len(nns)

# return whether or not model thinks a data_point earns above
def predict(data_point, training_set, k, t=False):
    tally = count(knn(data_point, training_set, k, test=t), k)
    # if at least half of the neighbors have an income >50k
    # predict 1
    return tally


# Q8
def cross_validation(k):
    # split array into 4 parts
    a, b, c, d = np.array_split(train_data, 4)
    results = []

    # test a
    training_set = np.concatenate((b, c, d))
    total = 0
    for row in a:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - train_data[id, -1])

    results.append(get_height(total/len(a)))

    # test b
    training_set = np.concatenate((a, c, d))
    total = 0
    for row in b:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - train_data[id, -1])

    results.append(get_height(total/len(b)))

    # test c
    training_set = np.concatenate((a, b, d))
    total = 0
    for row in c:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - train_data[id, -1])
    results.append(get_height(total/len(c)))

    # test d
    training_set = np.concatenate((a, b, c))
    total = 0
    for row in d:
        id = int(row[0])
        total += abs(predict(row, training_set, k) - train_data[id, -1])

    results.append(get_height(total/len(d)))

    return results

# Q9
# accuracy on training set when using the entire training set
def training_accuracy(k):
    total = 0
    for i in range(0, len(train_data)):
        total += abs(predict(train_data[i], train_data, k) - train_data[i, -1])

    return get_height(total / len(train_data))

def training_stats():
    test_arr = [1,2,4,8,16,32,64,128,len(train_data)]
    stats = []
    for K in test_arr:
        # get training accuracy
        trn_acc = training_accuracy(K)
        # get cross validation accuracies
        cv_accs = np.array(cross_validation(K))

        # keep track of stats for every iteration
        new = [K, get_height(trn_acc), get_height(cv_accs.mean()), get_height(cv_accs.var())]
        print(new)
        stats.append(new)

    np.savetxt("stats.csv", stats, delimiter=',', fmt='%f')






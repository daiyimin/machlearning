import pandas as pd
import numpy as np

# load raw data in csv file
# filename: csv file name
# Note:
#   the headline of csv is ignored
#   all columns but last one are used as x
#   the last column is used as y
# return
#   x,y: raw data in csv file
def load_csv(filename):
    df = pd.read_csv(filename)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    return (x, y)

# select train data and test data from raw data
# x,y: raw data
# percentage: the percentage of data to be used as train data
# return
#   x_train, y_train, x_test, y_test: train data and test data
def model_selection(x, y, percentage):
    num = len(x)
    sel_num = int(num * percentage)

    # generate an array of random int, the length is same as x data
    # 1000 shall be enough to generate fully scattered random int
    randomint_array = np.random.randint(0, 1000, num)
    # get the sorted indices of the random int array
    sorted_indices = np.argsort(randomint_array)
    # train_indices is the indices of biggest random int, and the size is sel_num
    # the value of train_indices is random, so use it to generate train data
    train_indices = sorted_indices[:sel_num]
    test_indices = sorted_indices[sel_num:]

    # generate train and test data
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train, x_test, y_test)

# rate classifier against given x and y
# return:
#   score, 0% ~ 100%
def rate_classifier(classifier, x, y):
    size = len(x)
    correct = 0
    for i in range(size):
        if classifier.predict(x[i]) == y[i]:
            correct += 1
    return float(correct / size)
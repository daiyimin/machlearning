import numpy as np
from StatisticLearning.KDTree import KDTree

class KNeighboursClassifier:
    def __init__(self, k = 1):
        self.k = k

    # make k neighbour classifier parameters fit to the train data
    # x_train: each line is a data set, each column is one feature in data set
    # y_train: each value is category of corresponding x_train data. The category value must be integers.
    # no return, the trained model parameters is updated in the perceptron instance
    def fit(self, x_train, y_train):
        # train KDTree
        self.kdt = KDTree()
        self.kdt.build(x_train)
        #self.x_train = x_train
        self.y_train = y_train

    # predict the category of test Xs
    def predict(self, test_xs):
        # search KDTree
        distances, neighbors = self.kdt.searchKNeighbours(test_xs, self.k)

        categories = []
        num_xs = len(test_xs)
        for i in range(0, num_xs):
            neighbour_categories = self.y_train[neighbors[i]]
            # count all neighbour categories
            count = np.bincount(neighbour_categories)
            # vote for the most popular neighbour category
            category = np.array([np.argmax(count)])

            categories.append(category)
        return np.array(categories)
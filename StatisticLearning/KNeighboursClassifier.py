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

    # predict the category of targets
    def predict(self, targets):
        # search KDTree
        distances, neighbors = self.kdt.searchKNeighbours(targets, self.k)

        flags = None
        num_target = len(targets)
        for i in range(0, num_target):
            neighbour_flags = self.y_train[neighbors[i]]
            # count all neighbour flags
            count = np.bincount(neighbour_flags)
            # use the neighbour flags appears most times as flag of target
            flag = np.array([np.argmax(count)])
            # concatenate flags of targets into a numpy array (i.e. flags)
            if flags is None:
                flags = flag
            else:
                flags = np.concatenate((flags, flag), axis=0)

        return flags
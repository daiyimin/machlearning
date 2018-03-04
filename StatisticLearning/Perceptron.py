import numpy as np

# This is a two category perceptron
class Perceptron:
    def __init__(self, n_iter = 100, eta = 0.1):
        self.n_iter = n_iter
        self.eta = eta

    # make perceptron parameters fit to the train data
    # x_train: each line is a data set, each column is one feature in data set
    # y_train: each value is category of corresponding x_train data. The value must be -1 and 1.
    # no return, the trained model parameters is updated in the perceptron instance
    def fit(self, x_train, y_train):
        # number of train data
        n_train = len(x_train)

        # calculate gram matrix of x_train
        gram = np.dot(x_train, x_train.T)

        # initialize alpha and b
        alpha = np.zeros(n_train)
        b = 0

        not_finished = True
        n_iter = 0
        while not_finished and (n_iter < self.n_iter):
            not_finished = False
            n_iter = n_iter + 1
            # in each iteration, go through all train data
            for i in range(n_train):
                err = y_train[i]*(np.dot(alpha*y_train, gram[:,i]) + b)
                if err <= 0:
                    # if this train data is wrongly categorized, adjust parameters of perceptron
                    alpha[i] = alpha[i] + self.eta
                    b = b + self.eta*y_train[i]
                    not_finished = True

        self.w = np.dot(alpha*y_train, x_train)
        self.b = b

    # predict the category of x
    def predict(self, x):
        if np.dot(self.w, x) + self.b > 0:
            return 1
        else:
            return -1
import numpy as np
import pandas as pd

class MaxEntroPyGIS:
    # iternate times
    iter = None

    def __init__(self, iter=1000):
        self.iter = iter

    def initialize_param(self, x_train, y_train):
        # unique categories
        self.categories = y_train.unique()
        # size of training data set
        self.data_num = len(x_train)

        # initialize C of GIS
        # in this implemenation, the shape of x data is same. Don't support variable x shape.
        self.C = x_train.shape[1]
        all_features = pd.Series()
        # pack all features in train data set into a big pandas Series
        for j in range(0, self.C):
            tmp_features = pd.Series(list(zip(x_train[j], y_train)))
            all_features = all_features.append(tmp_features)
        # calculate feature expectation on empirical distribution
        self.emp_epf = all_features.value_counts()/self.data_num
        # unique features, get from the axes[0] of feature expectation
        self.features = self.emp_epf.axes[0]
        # number of unique features
        self.feature_num = len(self.features)
        # initialize w of max entropy model
        self.w = pd.Series(np.zeros(self.feature_num))

    def train(self, x_train, y_train):
        self.initialize_param(x_train, y_train)

        for i in range(0, self.iter):
            epf = self.calculate_epf(x_train)
            last_w = self.w
            self.w = last_w + 1/self.C * np.log(self.emp_epf/epf)

            if sum((self.w - last_w) > 0.01) == 0:
                break

    # calculate a fi(x,y) matrix
    # return fi matrix, see example:
    # y\i   0           1           2           ......   i
    # y0    f0(x,y0)    f1(x,y0)    f2(x,y0)             fi(x,y0)
    # y1    f0(x,y1)    f1(x,y1)    f2(x,y1)             fi(x,y1)
    # ......
    def calculate_fi(self, x):
        # define a lambda function for calculate fi(x,y)
        fi = lambda i, x, y: 1 if self.features[i] in [(xj, y) for xj in x] else 0
        # calculate fi(x,y) matrix
        fi_matrix = pd.DataFrame()
        for y in self.categories:
            fi_row = []
            for i in range(0, self.feature_num):
                fi_val = fi(i, x, y)
                fi_row.append(fi_val)
            fi_matrix = fi_matrix.append(pd.Series(fi_row), ignore_index=True)

        return fi_matrix

    # calculate probability of each category
    # return a matrix of category and its probability, see example:
    # index\    y       py
    # 0         y0      p(y0)
    # 1         y1      p(y1)
    # ....
    def calculate_prob(self, fi_matrix):
        Z = 0
        for r in range(0, len(fi_matrix)):
            Z += np.exp(sum(fi_matrix.loc[r] * self.w))

        prob = pd.DataFrame()
        for r in range(0, len(self.categories)):
            y = self.categories[r]
            py = np.exp(sum(fi_matrix.loc[r] * self.w)) / Z
            prob = prob.append(pd.Series([y, py]), ignore_index=True)

        return prob

    def calculate_epf(self, x_train):
        epf = np.zeros(self.feature_num)
        for i in range(0, len(x_train)):
            fi_matrix = self.calculate_fi(x_train.loc[i])
            prob = self.calculate_prob(fi_matrix)
            py = prob[1]
            # sum(1/N *p(y|x)*fi(y,x)), p(x) = 1/N
            epf += 1 / self.data_num * np.dot(py, fi_matrix)
        return epf

    def predict(self, x):
        fi_matrix = self.calculate_fi(x)
        prob = self.calculate_prob(fi_matrix)
        return prob
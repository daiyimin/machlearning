import numpy as np
from StatisticLearning.Models.CartRegressionTree import CartRegressionTree

class DecisionTreeRegressor:
    def rate_tree(self, tree, x_test, y_test):
        y_pred = []
        for x in x_test:
            y = tree.predict(x)
            y_pred.append(y)

        sum = 0
        for y1, y2 in zip(y_pred, y_test):
            sum += np.square(y1 - y2)
        return sum

    def fit(self, x, y):
        tree = CartRegressionTree(True)
        # use 90% as train data, 10% as validate data
        train_num = int(len(x)*0.9)
        # get train data
        # do transform because CartTree requires Pandas data model
        x_train = x[:train_num]
        y_train = y[:train_num]

        tree = CartRegressionTree(True)
        tree.build(x_train, y_train)
        T = tree.post_prune()

        # get validate data
        x_valid = x[train_num:]
        y_valid = y[train_num:]
        sqr_err = np.inf
        sel_t = None
        # select a optimal tree which gets minimal error on validate data
        # iterate through all trees in list T
        for t in T.values():
            score = self.rate_tree(t, x_valid, y_valid)
            # save the tree that gets max score
            if score < sqr_err:
                sqr_err = score
                sel_t = t

        self.tree = sel_t

    # xs is np.array type
    def predict(self, xs, tree=None):
        predictions = []
        # predict one by one in xs, and put in a list
        for x in xs:
            prediction = self.tree.predict(x)
            predictions.append(prediction)
        return np.array(predictions)


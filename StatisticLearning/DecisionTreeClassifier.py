import numpy as np
import pandas as pd
from StatisticLearning.Models.CartTree import CartTree

class DecisionTreeClassifier:
    # rate the predict precision
    # return a percentage score
    # y_preds, y are np.array
    def rate_tree(self, tree, x_test, y_test):
        y_pred = []
        for x in x_test:
            y = tree.predict(x)
            y_pred.append(y)
        correct = sum(y_pred == y_test)
        total = len(y_pred)
        return float(correct / total)

    # xs is np.array type
    def predict(self, xs):
        categories = []
        # predict one by one in xs, and put in a list
        for x in xs:
            # Attention: CartTree.predict requires Pandas.Series as parameter.
            category = self.tree.predict(pd.Series(x))
            categories.append(category)
        return np.array(categories)

    # x, y are np.array type
    def fit(self, x, y):
        # use 90% as train data, 10% as validate data
        train_num = int(len(x)*0.9)
        # get train data
        # do transform because CartTree requires Pandas data model
        x_train = pd.DataFrame(x[:train_num])
        y_train = pd.Series(y[:train_num])
        tree = CartTree()
        # build tree
        tree.build(x_train, y_train)
        # post prune
        T = tree.post_prune()

        # get validate data
        x_valid = x[train_num:]
        y_valid = y[train_num:]

        max_score = -np.inf
        sel_t = None
        # select a optimal tree which gets max score on validate data
        # iterate through all trees in list T
        for t in T.values():
            # use t to do predict
            # y_preds = self.predict(x_valid, t)
            # get the score
            # score = self.rate_tree(y_preds, y_valid)
            score  = self.rate_tree(t, x_valid, y_valid)
            # save the tree that gets max score
            if score > max_score:
                max_score = score
                sel_t = t
        # save the tree
        self.tree = sel_t

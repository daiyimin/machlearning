import numpy as np
import pandas as pd
from StatisticLearning.Models.CartTree import CartTree

class DecisionTreeClassifier:
    # rate the predict precision
    # return a percentage score
    # y_preds, y are np.array
    def rate_tree(self, y_preds, y):
        correct = sum(y_preds == y)
        total = len(y_preds)
        return float(correct / total)

    # xs is np.array type
    def predict(self, xs, tree=None):
        # during post prune, predict method is called with a given tree
        # in that case, don't use self.tree to do predict
        # in other cases(called by end users), use self.tree to do predict
        if tree is None:
            tree = self.tree

        categories = []
        # predict one by one in xs, and put in a list
        for x in xs:
            # Attention: CartTree.predict requires Pandas.Series as parameter.
            category = tree.predict(pd.Series(x))
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
            y_preds = self.predict(x_valid, t)
            # get the score
            score = self.rate_tree(y_preds, y_valid)
            # save the tree that gets max score
            if score > max_score:
                max_score = score
                sel_t = t
        # save the tree
        self.tree = sel_t

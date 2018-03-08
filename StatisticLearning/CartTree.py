import numpy as np
import pandas as pd

# DecisionTree implemented by CART algorithm
class CartTree:
    # threshold of gini index for stop creating new tree
    gini_threshold = 0.01
    # threshold of train data number for stop creating new tree
    train_num_threshold = 1
    # list of child, children[0] is left child, children[1] is right child
    children = None
    # gini index of this tree/leaf
    gini_index = None
    # category of this tree/leaf
    category = None
    # selected feature of this tree
    sel_feature = None
    # selected feature value used to slice x data for creating children
    sel_feature_value = None

    def __init__(self):
        self.children = list()

    # helper function to calculate gini index
    # data: data used to calculate gini index. It's a pandas Series.
    def gini(self, data):
        total_count = data.count()
        value_counts = data.value_counts()
        gini_index = 1 - sum(np.square(value_counts/total_count))
        return gini_index

    def build(self, x, y):
        # get gini index of y, saved as gini index of this tree/leaf
        self.gini_index = self.gini(y)
        # use the most frequent y value as category of this tree/leaf
        self.category = y.value_counts().keys()[0]
        # number of train data in x
        train_x_num = x.shape[0]  # same as len(x)

        # 1. if gini_index is small, it means the uncertainty of this tree is small. Then it's
        # senseless to create more children. Leave this node as leaf node.
        # 2. if train data number is small, it means the overhead of create more children is
        # not valuable. Leave this node as leaf node.
        if self.gini_index < self.gini_threshold or train_x_num < self.train_num_threshold:
            return

        sel_feature, sel_feature_value, min_gini = None, None, np.inf
        # iterate through all feature columns in x
        for feature in x.columns:
            # if feature is already used by a parent tree, don't use it again.
            if feature == "used":
                continue
            # get the feature column xj
            xj = x[feature]
            # get unique values in column xj
            unique_xj = xj.unique()
            # iterate through all unique values of xj
            for ajl in unique_xj:
                y1 = y[xj == ajl]
                y1_ratio = len(y1) / train_x_num
                y2 = y[xj != ajl]
                y2_ratio = len(y2) / train_x_num
                # calculate Gini(D|A) in (5.25)
                gini = y1_ratio * self.gini(y1) + y2_ratio * self.gini(y2)
                if gini < min_gini:
                    # update min_gini
                    min_gini = gini
                    # update selected feature, selected feature value
                    self.sel_feature, self.sel_feature_value = feature, ajl
                # if xj has 2 unique values, because their gini index are same,
                # only calculate gini index for the 1st value. Skip rest values.
                if len(unique_xj) == 2:
                    break;

        # if min_gini will not be updated, it means all feature are "used"
        # in this case, don't bother to create more children
        if min_gini == np.inf:
            return

        # get a pd.Series containing selected feature of all x
        sel_x = x[self.sel_feature]
        # calculate the slice for left child and right child
        left_slice = sel_x.isin([self.sel_feature_value])
        right_slice = sel_x.isin([self.sel_feature_value]) == False
        # mark the feature is used
        x.rename(columns={self.sel_feature: 'used'}, inplace=True)
        for slice in left_slice, right_slice:
            tree = CartTree()
            x_slice = x[slice]
            y_slice = y[slice]
            tree.build(x_slice, y_slice)
            self.children.append(tree)
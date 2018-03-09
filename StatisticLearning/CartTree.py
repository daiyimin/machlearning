import numpy as np
import copy

# DecisionTree implemented by CART algorithm
class CartTree:
    # threshold of gini index for stop creating new tree
    gini_threshold = 0.01
    # threshold of train data number for stop creating new tree
    train_num_threshold = 1
    # list of child, children[0] is left child, children[1] is right child
    children = None
    # number of x of this tree
    x_num = None
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

    # x: train x data in format of pandas.DataFrame
    # y: train y data in format of pandas.Series
    def build(self, x, y):
        # get gini index of y, saved as gini index of this tree/leaf
        self.gini_index = self.gini(y)
        # use the most frequent y value as category of this tree/leaf
        self.category = y.value_counts().keys()[0]
        # number of train data in x
        self.x_num = x.shape[0]  # same as len(x)

        # 1. if gini_index is small, it means the uncertainty of this tree is small. Then it's
        # senseless to create more children. Leave this node as leaf node.
        # 2. if train data number is small, it means the overhead of create more children is
        # not valuable. Leave this node as leaf node.
        if self.gini_index < self.gini_threshold or self.x_num < self.train_num_threshold:
            return

        sel_feature, sel_feature_value, min_gini = None, None, np.inf
        # iterate through all feature columns in x
        for feature in x.columns:
            # if feature is already used by a parent tree, don't use it again.
            if feature == "used":
                continue
            # get the feature column xj, j represents column index
            xj = x[feature]
            # get unique values in column xj
            unique_xj = xj.unique()
            # iterate through all unique values of xj
            for ajl in unique_xj:
                y1 = y[xj == ajl]
                y1_ratio = len(y1) / self.x_num
                y2 = y[xj != ajl]
                y2_ratio = len(y2) / self.x_num
                # calculate Gini(D|A) in (5.25)
                gini = y1_ratio * self.gini(y1) + y2_ratio * self.gini(y2)
                if gini < min_gini:
                    # update min_gini
                    min_gini = gini
                    # update selected feature, selected feature value
                    self.sel_feature, self.sel_feature_value = feature, ajl
                # if xj has 2 unique values, because their gini index and way of slice are same,
                # only calculate gini index for the first value. Skip the other..
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

    def post_prune_done(self, depth=0):
        if depth >= 2:
            return False
        for child in self.children:
            if child.post_prune_done(depth+1):
                return False
        return True

    def calculate_alpha(self, alpha=np.inf):
        if len(self.children) == 0:
            # if this is a leaf node
            # calculate Ct, Ct = Nt*Gt(T)
            Ct = self.x_num * self.gini_index
            # 1 represent one leaf node
            return Ct, 1, alpha
        else:
            # if this is a non-leaf node
            # calculate Ct, Ct = Nt*Gt(T)
            Ct = self.x_num * self.gini_index
            CTt, leaf_num = 0, 0
            # calculate CTt, leaf_num
            for child in self.children:
                # call calculate_alpha on all children
                child_Ct, child_leaf_num, alpha = child.calculate_alpha(alpha)
                # sum up child_Ct to calculate CTt
                CTt += child_Ct
                # sum up leaf numbers in all children
                leaf_num += child_leaf_num
            # calculate gt
            self.gt = (Ct - CTt) / (leaf_num - 1)
            # if gt is less than alpha, update alpha
            if self.gt < alpha:
                alpha = self.gt
            # return CTt，leaf number of this tree, as well as alpha
            return CTt, leaf_num, alpha

    def prune(self, alpha):
        if len(self.children) != 0:
            # if this is not a leaf node, check gt equals alpha or not
            if self.gt == alpha:
                # if gt equals to alpha, do prune on this node!
                self.children.clear()
                # return True to finish prune method
                return True
            else:
                # if gt doesn't equal to alpha, try to prune its children
                for child in self.children:
                    # call prune method on each child
                    if child.prune(alpha):
                        # if prune is done in a child, return Ture to finish prune method.
                        return True
        return False

    def post_prune(self):
        T = {}
        k = 0
        T[k] = copy.deepcopy(self)
        while not self.post_prune_done():
            CTt, leaf_num, alpha = self.calculate_alpha()
            self.prune(alpha)
            k += 1
            T[k] = copy.deepcopy(self)

        return T
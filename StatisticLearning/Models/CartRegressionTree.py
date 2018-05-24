import numpy as np
import copy

class CartRegressionTree:

    def __init__(self, root = False):
        self.root = root
        self.children = list()

    def square_err(self, data):
        mean = np.mean(data)
        result = sum(np.square(data - mean))
        return result

    def build(self, x, y):
        self.sqr_err = self.square_err(y)
        self.x_num = x.shape[0]
        self.feature_num = x.shape[1]
        self.y = y
        self.mean = np.mean(y)

        if self.sqr_err < 0.01 and self.x_num <= 1:
            return

        self.j, self.s = None, None
        sqr_err = self.sqr_err
        for j in range(self.feature_num):
            for s in range(self.x_num):
                s_val = x[s,j]

                y1 = y[x[:, j] <= s_val]
                y2 = y[x[:, j] > s_val]
                tmp_err = self.square_err(y1) + self.square_err(y2)
                if tmp_err < sqr_err:
                    sqr_err = tmp_err
                    self.j, self.s = j, s

        if self.j == None:
            return

        self.s_val = x[self.s, self.j]
        left_slice = x[:, self.j] <= self.s_val
        right_slice = x[:, self.j] > self.s_val

        for slice in left_slice, right_slice:
            tree = CartRegressionTree()
            x_slice = x[slice]
            y_slice = y[slice]
            tree.build(x_slice, y_slice)
            self.children.append(tree)

        return

    def leaf_num(self):
        if len(self.children) == 0:
            return 1

        sum = 0
        for child in self.children:
            sum += child.leaf_num()
        return sum

    def post_prune_completed(self, depth=0):
        # depth first search cart tree, if tree depth is more than 2, return False (not completed)
        if depth >= 2:
            return False
        for child in self.children:
            if not child.post_prune_completed(depth+1):
                return False
        # if tree depth is less than 2, return True (completed)
        return True

    # calculate alpha=min(gt) for cart tree
    def calculate_alpha(self, alpha=np.inf):
        if len(self.children) == 0:
            # this is a leaf node
            # calculate Ct, Ct = sqr_err
            Ct = self.sqr_err
            # 1 represent one leaf node
            return Ct, 1, alpha
        else:
            # this is a non-leaf node t. Tt is the sub-tree whose root node is t.
            # calculate Ct, Ct = sqr_err
            Ct = self.sqr_err
            # calculate CTt, leaf_num
            CTt, leaf_num = 0, 0
            for child in self.children:
                # call calculate_alpha on all children
                child_Ct, child_leaf_num, alpha = child.calculate_alpha(alpha)
                # sum up child_Ct to calculate CTt
                CTt += child_Ct
                # sum up leaf numbers in all children
                leaf_num += child_leaf_num

            if self.root:
                # root tree will not be pruned. So don't bother to calculate gt for it.
                # set root.gt to np.inf, so it never gets pruned.
                self.gt = np.inf
                # return alpha
                return alpha
            else:
                # calculate gt of Tt
                self.gt = (Ct - CTt) / (leaf_num - 1)
                # if gt is less than alpha, update alpha
                if self.gt < alpha:
                    alpha = self.gt
                # return to parent node
                # CTt，leaf number of this tree, alpha
                return CTt, leaf_num, alpha

    # prune cart tree using alpha
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
        # save root tree
        T[k] = copy.deepcopy(self)

        while not self.post_prune_completed():
            # calculate alpha for current cart tree
            alpha = self.calculate_alpha()
            # prune current cart tree using alpha
            self.prune(alpha)
            # save the pruned tree
            k += 1
            T[k] = copy.deepcopy(self)

        # return list of pruned tree
        return T

    def is_leaf(self):
        return len(self.children) == 0

    def predict(self, x):
        node = self
        # make decisions according to x feature values till meet a leaf node
        while not node.is_leaf():
            if x[node.j] <= node.s_val:
                # if value <= node.s_val, go to children[0]
                # this is decided during building the tree
                node = node.children[0]
            else:
                node = node.children[1]
        # return mean of the leaf node
        return node.mean

    # this function is used by two class boosting decision tree
    # refer to Algorithm 5 in https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451
    def calculate_gamma(self):
        if self.is_leaf():
            prediction = np.sum(self.y) / np.sum(abs(self.y) * (2-abs(self.y)))
            self.mean = prediction
            return

        for child in self.children:
            child.calculate_gamma()
import numpy as np
from StatisticLearning.Models.CartRegressionTree import CartRegressionTree

'''
Gradient Boosting Decision Tree 
For all the steps of this algorithm, refer to Two-class logistic regression and classification in https://blog.csdn.net/yc1203968305/article/details/78312166
'''
class GBDT:
    # how many leaves for each boosting tree
    J = None
    # iteration number of the algorithm
    M = None
    # list of boosting trees
    boosting_trees = []

    def __init__(self, num_tree_leaves, num_iterations=100):
        self.J = num_tree_leaves
        self.M = num_iterations

    def build_tree(self, X_train, Y_train):
        # train a new CART regression tree
        tree = CartRegressionTree(True)
        tree.build(X_train, Y_train)
        T = tree.post_prune()
        # get a J leaves tree in the post prune tree list
        sel_t = T[0]
        for t in T.values():
            if t.leaf_num() >= self.J:
                sel_t = t
            else:
                break
        return sel_t

    def loss_func_gradient(self, X_train, Y_train):
        # step 3: calculate the gradient of Loss Function
        N = len(X_train)
        y = []
        for i in range(N):
            yi = 2 * Y_train[i] / (1 + np.exp(2 * Y_train[i] * self.predict(X_train[i])))
            y.append(yi)
        gradient = np.array(y)

        return gradient

    def train(self, X_train, Y_train, print_cost=False):
        N = len(X_train)

        # step 1: calculate F0
        y_mean = np.mean( Y_train )
        self.F0 = 0.5 * np.log((1 + y_mean) / (1 - y_mean))

        # step 2: looping M times
        for m in range(self.M):
            # step 3: calculate the gradient of Loss Function
            pseoudo_Y = self.loss_func_gradient(X_train, Y_train)

            # step 4: train a new decision tree
            tree = self.build_tree(X_train, pseoudo_Y)
            # step 5: calculate gama of tree leaves
            tree.calculate_gamma()

            # step 6: append tree to additive model
            self.boosting_trees.append(tree)

    # calculate P+ using FmX
    # https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451
    # refer to equation (21)
    def log_odds(self, Fx):
        p_positive = 1 /(1+ np.exp(-2*Fx))
        return p_positive

    def predict(self, X):
        # sum up prediction of all boosting trees
        FmX = self.F0
        for tree in self.boosting_trees:
             FmX += tree.predict(X)

        p_positive = self.log_odds(FmX)
        return p_positive

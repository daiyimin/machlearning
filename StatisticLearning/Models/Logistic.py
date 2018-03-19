import numpy as np

class Logistic:
    num_iterations = None
    learning_rate = None
    w = None    # weight of Logistic model
    b = None    # bias of Logistic model

    def __init__(self, num_iterations=2000, learning_rate=0.1):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        s = 1.0 / (1 + 1 / np.exp(z))
        return s

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above
        Arguments:
        w -- weights, a numpy array of size (feature number, 1)
        b -- bias, a scalar
        X -- data of size (number of examples, feature number)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (number of examples, 1)
        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        """

        m = X.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        # https://www.missshi.cn/api/view/blog/59aa08fee519f50d04000170
        A = self.sigmoid(np.dot(X, w) + b)  # compute activation
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))  # compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = 1.0 / m * np.dot((A - Y), X)
        db = 1.0 / m * np.sum(A - Y)
        cost = np.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def train(self, X_train, Y_train, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        Arguments:
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        print_cost -- True to print the loss every 100 steps
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """

        costs = []

        w = np.zeros(X_train.shape[1])
        b = 0
        for i in range(self.num_iterations):  # 每次迭代循环一次， num_iterations为迭代次数

            # Cost and gradient calculation
            grads, cost = self.propagate(w, b, X_train, Y_train)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        # save model parameter
        self.w = w
        self.b = b

    def calculate_prob(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        prob = self.sigmoid(np.dot(X, self.w) + self.b)

        # prob[0, i] is the probability of X[i] = 1. If it is greater than 0.5, X[i] is most probably to be 1.
        return prob
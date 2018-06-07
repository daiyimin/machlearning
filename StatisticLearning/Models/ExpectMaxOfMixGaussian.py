import numpy as np
from scipy.stats import multivariate_normal

class ExpectationMaximumOfMixtureGaussian:

    # This class implement EM algorithm of Mixture Gaussian Models. Each Gaussian Model obeys same multivariate normal distribution.
    # The output of a multivariate Gaussian Model is a D variant vector(array).
    # K: the number of Gaussian models
    def __init__(self, K, iterations=1000):
        self.K = K
        self.iterations = iterations

    # Y is the observation value matrix. Each line is an observation.
    # N = Y.shape[0] is the number of observation value
    # D = Y.shape[1] is dimension of an observation value.
    def train(self, Y):
        N = Y.shape[0]
        D = Y.shape[1]

        # initialize coefficients of mixture Gaussian models
        self.alpha = np.ones(self.K)
        # initialize the parameter of mixture Gaussian models
#        self.mean = np.zeros((self.K, D))   # mean[k] is mean of k-th Gaussian model, k = [0, K)
        self.mean = np.random.random(size=(self.K, D))
        self.cov = np.zeros((self.K, D, D))
        self.cov[:] = np.eye(D)   # cov[k] is covariance of k-th Gaussian model, k = [0, K)

        #self.mean = np.array([[-1,0],[1,0]])

        L = -np.Inf
        for i in range(self.iterations):
            # calculate gama according to P164.
            gama = np.zeros((N, self.K))
            # E step
            for j in range(N):
                w = np.zeros(self.K)
                for k in range(self.K):
                    w[k] = self.alpha[k] * multivariate_normal.pdf(Y[j], self.mean[k], self.cov[k])
                sum = np.sum(w)
                for k in range(self.K):
                    gama[j,k] = w[k] / sum

            # M step
            # update alpha, 9.32
            self.alpha = np.sum(gama, axis=0) / N
            # update covariance, 9.31
            for k in range(self.K):
                cov = np.zeros((D, D))
                for j in range(N):
                    delta = Y[j:j+1,:] - self.mean[k:k+1,:]
                    cov += gama[j,k] * delta.T * delta
                self.cov[k] = cov / np.sum(gama[:, k])
            # update mean, 9.30
            self.mean = ( np.dot(Y.T, gama) / np.sum(gama, axis=0) ).T

            newL = self.likehood(Y)
            if newL - L  < 0.001 :
                break
            else:
                L = newL


    def likehood(self, Y):
        N = Y.shape[0]

        L = 0
        for j in range(N):
            # calculate likehood of Y[j]
            Lj = 0
            for k in range(self.K):
                Lj += self.alpha[k] * multivariate_normal.pdf(Y[j], self.mean[k], self.cov[k])

            # sum up likehood of all Y
            L += np.log(Lj)

        return L

    # based on the initial parameters, the sequence of learnt gaussian model is not predictable.
    # for example, the mean[0] and cov[0] could belong to 2nd model, while the mean[1] and cov[1] could belong to 1st model
    # therefore, the meaning of category in prediction result is changing
    def predict(self, y):
        prob = []
        for k in range(self.K):
            prob.append( multivariate_normal.pdf(y, self.mean[k], self.cov[k]) )

        sort_idx = np.argsort(prob)
        category = sort_idx[-1]
        return category
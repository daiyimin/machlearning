import numpy as np
from StatisticLearning.Models.ExpectMaxOfMixGaussian import ExpectationMaximumOfMixtureGaussian

class ExpectMaxOfMixGaussianClassifier:
    def __init__(self, K):
        self.K = K

    def fit(self, Y):
        self.em = ExpectationMaximumOfMixtureGaussian(self.K)
        self.em.train(Y)

    def predict(self, Y):
        categories = []
        for y in Y:
            categories.append(self.em.predict(y))
        return np.array(categories)
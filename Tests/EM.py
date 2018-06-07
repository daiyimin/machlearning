from StatisticLearning import ExpectationMaximumOfMixtureGaussian
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import warnings

class TestGDBTClassifier(unittest.TestCase):
    def test_GDBTClassifier_with_iris(self):
        warnings.filterwarnings('ignore')
        mean = (-2, -1)
        cov = [[1, -0.5], [-0.3, 1]]
        x = np.random.multivariate_normal(mean, cov, 100)

        mean = (4, 3)
        cov = [[2,1], [0.4, 1]]
        y = np.random.multivariate_normal(mean, cov, 100)

        z = np.vstack((x, y))

        em = ExpectationMaximumOfMixtureGaussian(2)
        em.train(z)

        def f(x,y, mean, cov):
            lines = []
            for lx, ly in zip(x,y):
                cols = []
                for cx, cy in zip(lx, ly):
                    value = multivariate_normal.pdf((cx,cy), mean, cov)
                    cols.append(value)
                lines.append(cols)
            return np.array(lines)

        x1 = np.linspace(-10, 10, 50)
        y1 = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x1, y1)
        plt.contour(X, Y, f(X, Y,em.mean[0], em.cov[0]))
        plt.contour(X, Y, f(X, Y,em.mean[1], em.cov[1]))
        plt.plot(x[:,0],x[:,1], 'ro', color='red')
        plt.plot(y[:,0],y[:,1], 'ro', color='green')
        plt.show()


if __name__ == '__main__':
    unittest.main()
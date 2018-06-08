from StatisticLearning import ExpectMaxOfMixGaussianClassifier
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from Tests import TestUtils

class TestEMClassifier(unittest.TestCase):
    def genenrate_model1_data(self, n):
        mean = (-2, -1)
        cov = [[1, -0.5], [-0.5, 1]]
        Y = np.random.multivariate_normal(mean, cov, n)
        return Y

    def genenrate_model2_data(self, n):
        mean = (1, 3)
        cov = [[2,0.4], [0.4, 1]]
        Y = np.random.multivariate_normal(mean, cov, n)
        return Y

    def genenrate_model3_data(self, n):
        mean = (-3, 1)
        cov = [[1,0.2], [0.2, 1]]
        Y = np.random.multivariate_normal(mean, cov, n)
        return Y

    def plot(self, model_data, learnt_model):
        def f(x,y, mean, cov):
            lines = []
            for lx, ly in zip(x,y):
                cols = []
                for cx, cy in zip(lx, ly):
                    value = multivariate_normal.pdf((cx,cy), mean, cov)
                    cols.append(value)
                lines.append(cols)
            return np.array(lines)

        # draw model contour
        x1 = np.linspace(-10, 10, 50)
        y1 = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x1, y1)

        for i in range(len(learnt_model.alpha)):
            C = plt.contour(X, Y, f(X, Y,learnt_model.mean[i], learnt_model.cov[i]))
            plt.clabel(C, inline=True, fontsize=8)

        style = ['b.','r+','gx','yo']
        for i in range(len(model_data)):
            data = model_data[i]
            plt.plot(data[:, 0], data[:, 1], style[i])

        plt.show()

    def test_EMClassifier(self):
        # generate train data
        y1 = self.genenrate_model1_data(100)
        y2 = self.genenrate_model2_data(200)
        Y = np.vstack((y1, y2))

        # train EM classifier
        emc = ExpectMaxOfMixGaussianClassifier(2)
        emc.fit(Y)

        # draw train data and contour of learnt model together
        learnt_model = emc.em
        #self.plot([y1, y2], learnt_model)

        # we know in the input data Y, model1 data accounts for 1/3, model2 data accounts for 2/3
        # so in EM algorithm, if coefficient of 1st learnt model is closer to 1/3，EM category of model1 is 0, and EM category of model2 is 1
        # if coefficient of 1st learnt model is closer to 2/3，EM category of model2 is 0, EM category of model1 is 1
        if abs(learnt_model.alpha[0] - 1/3) < abs(learnt_model.alpha[1] - 1/3):
            model1_category, model2_category = 0,1
        else:
            model1_category, model2_category = 1,0

        # generate test data
        y1 = self.genenrate_model1_data(25)
        y2 = self.genenrate_model2_data(25)
        Y = np.vstack((y1, y2))
        score = TestUtils.rate_batch_classifier(emc, Y, [model1_category for i in y1] + [model2_category for i in y2])
        print("test_EMClassifier", score)
        self.assertTrue(score > 0.95)

    # def test_EMClassifier2(self):
    #     # generate train data
    #     y1 = self.genenrate_model1_data(50)
    #     y2 = self.genenrate_model2_data(100)
    #     y3 = self.genenrate_model3_data(150)
    #     Y = np.vstack((y1, y2, y3))
    #
    #     # train EM classifier
    #     emc = ExpectMaxOfMixGaussianClassifier(3)
    #     emc.fit(Y)
    #
    #     # draw train data and contour of learnt model together
    #     learnt_model = emc.em
    #     self.plot([y1, y2, y3], learnt_model)
    #
    #     def category(ratio, learnt_model):
    #         model_number = len(ratio)
    #         label = np.zeros(model_number)
    #
    #         for i in range(model_number):
    #             delta = []
    #             for j in range(model_number):
    #                 delta.append(abs(learnt_model.alpha[j] - ratio[i]))
    #             sorted_idx = np.argsort(delta)
    #             # the index model which has minimal delta is chosen as category
    #             label[i] = sorted_idx[0]
    #         return label
    #
    #     labels = category([1/6, 1/3, 1/2], learnt_model)
    #
    #     # generate test data
    #     y1 = self.genenrate_model1_data(25)
    #     y2 = self.genenrate_model2_data(25)
    #     y3 = self.genenrate_model2_data(25)
    #     Y = np.vstack((y1, y2, y3))
    #     score = TestUtils.rate_batch_classifier(emc, Y, [labels[0] for i in y1] + [labels[1] for i in y2] + [labels[2] for i in y3])
    #     print("test_EMClassifier", score)
    #     self.assertTrue(score > 0.95)

if __name__ == '__main__':
    unittest.main()
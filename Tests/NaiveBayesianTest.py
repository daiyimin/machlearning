from StatisticLearning import NaiveBayesianClassifier
from Tests import TestUtils
import numpy as np
import unittest

class TestNaiveBayesianClassifier(unittest.TestCase):

    def test_NaiveBayesianClassifier_with_MLE_0(self):
        x, y = TestUtils.load_iris_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        nb = NaiveBayesianClassifier(0) #set laplace=0 to use MLE
        nb.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(nb, x_test, y_test)
        print("test_NaiveBayesianClassifier_with_MLE_0", score)
        self.assertTrue(score > 0.80)

    def test_NaiveBayesianClassifier_with_BE_0(self):
        x, y = TestUtils.load_iris_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        nb = NaiveBayesianClassifier(1) #set laplace=0 to use BE
        nb.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(nb, x_test, y_test)
        print("test_NaiveBayesianClassifier_with_BE_0", score)
        self.assertTrue(score > 0.70)

    def test_NaiveBayesianClassifier_with_MLE_1(self):
        x, y = TestUtils.load_csv("TestData/car.data")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        nb = NaiveBayesianClassifier(0) #set laplace=0 to use MLE
        nb.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(nb, x_test, y_test)
        print("test_NaiveBayesianClassifier_with_MLE_1", score)
        self.assertTrue(score > 0.80)

    def test_NaiveBayesianClassifier_with_BE_1(self):
        x, y = TestUtils.load_csv("TestData/car.data")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        nb = NaiveBayesianClassifier(1) #set laplace=0 to use BE
        nb.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(nb, x_test, y_test)
        print("test_NaiveBayesianClassifier_with_BE_1", score)
        self.assertTrue(score > 0.80)

    def test_Chap04_with_MLE(self):
        x, y = TestUtils.load_csv("TestData/NaiveBayesianData.csv")

        nb = NaiveBayesianClassifier(0)
        nb.fit(x.astype(str), y)

        tx = np.array([[2,'S']])
        ty = np.array([-1])
        score = TestUtils.rate_batch_classifier(nb, tx, ty)
        print("test_Chap04_with_MLE", score)
        self.assertTrue(score > 0.80)

    def test_Chap04_with_BE(self):
        x, y = TestUtils.load_csv("TestData/NaiveBayesianData.csv")

        nb = NaiveBayesianClassifier(0.5)
        nb.fit(x.astype(str), y)

        tx = np.array([[2,'S']])
        ty = np.array([-1])
        score = TestUtils.rate_batch_classifier(nb, tx, ty)
        print("test_Chap04_with_BE", score)
        self.assertTrue(score > 0.80)

if __name__ == '__main__':
    unittest.main()
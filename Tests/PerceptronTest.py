from StatisticLearning import PerceptronClassifier
from Tests import TestUtils
import unittest

class TestPerceptron(unittest.TestCase):
    def test_perceptron(self):
        perceptron = PerceptronClassifier(100)
        x, y = TestUtils.load_perceptron_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.7)
        perceptron.fit(x_train, y_train)
        score = TestUtils.rate_classifier(perceptron, x_test, y_test)
        self.assertTrue(score > 0.99)

    def test_chap02(self):
        # Run the example in Chap02
        perceptron = PerceptronClassifier(100)
        x, y = TestUtils.load_csv("TestData/PerceptionData.csv")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.7)
        perceptron.fit(x, y)
        score = TestUtils.rate_classifier(perceptron, x_test, y_test)
        self.assertTrue(score > 0.99)

if __name__ == '__main__':
    unittest.main()
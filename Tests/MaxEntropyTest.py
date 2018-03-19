from StatisticLearning import MaxEntropyClassifier
from Tests import TestUtils
import unittest
import pandas as pd

class TestMaxEntropy(unittest.TestCase):
    def test_MaxEntropyClassifier_with_car(self):
        classifier = MaxEntropyClassifier()
        x, y = TestUtils.load_csv("TestData/car.data")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.9)
        classifier.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(classifier, x_test, y_test)
        print("test_MaxEntropyClassifier_with_car", score)
        self.assertTrue(score > 0.70)

    def test_MaxEntropyClassifier_with_weather(self):
        classifier = MaxEntropyClassifier()
        x, y = TestUtils.load_csv("TestData/MaxEnt.data")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 1)
        classifier.fit(x_train, y_train)

        xv = pd.Series(["Cloudy"])
        result = classifier.predict(xv)
        print(result)

        xv = pd.Series(["Sunny"])
        result = classifier.predict(xv)
        print(result)


if __name__ == '__main__':
    unittest.main()
from StatisticLearning import LogisticClassifier
from Tests import TestUtils
import unittest
import numpy as np
import warnings


class TestMaxEntropy(unittest.TestCase):
    def test_LogisticClassifier_with_iris(self):
        warnings.filterwarnings('ignore')

        classifier = LogisticClassifier()
        x, y = TestUtils.load_binary_class_iris_data(1)
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.8)
        classifier.fit(x_train, y_train)

        score = TestUtils.rate_batch_classifier(classifier, x_test, y_test)
        print("test_LogisticClassifier_with_iris", score)
        self.assertTrue(score >= 0.90)

if __name__ == '__main__':
    unittest.main()
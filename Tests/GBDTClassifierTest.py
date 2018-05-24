from StatisticLearning import CartTree
from StatisticLearning import GBDTClassifier
from Tests import TestUtils
import unittest
import copy
import warnings

class TestGDBTClassifier(unittest.TestCase):
    def test_GDBTClassifier_with_iris(self):
        warnings.filterwarnings('ignore')

        x, y = TestUtils.load_binary_class_iris_data(category=0)
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        classifier = GBDTClassifier()
        classifier.fit(x_train, y_train)
        score = TestUtils.rate_batch_classifier(classifier, x_test, y_test)
        print("test_GDBTClassifier_with_iris", score)
        # The accuracy of boosting tree is more than 99.99%! If compare it with other classifiers, it's amazing.
        self.assertTrue(score > 0.9999)

if __name__ == '__main__':
    unittest.main()
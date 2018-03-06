from StatisticLearning import NaiveBayesianClassifier
from StatisticLearning import Utils
from Tests import TestUtils
import unittest

class TestNaiveBayesianClassifier(unittest.TestCase):

    def test_NaiveBayesianClassifier(self):
        x, y = TestUtils.load_knn_data()
        x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.90)

        nb = NaiveBayesianClassifier()
        nb.fit(x, y)

        score = Utils.rate_batch_classifier(nb, x_test, y_test)
        print(score)
        self.assertTrue(score > 0.80)

if __name__ == '__main__':
    unittest.main()
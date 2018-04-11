from StatisticLearning import SVM
from Tests import TestUtils
import unittest

class TestSVM(unittest.TestCase):
    def test_SVM(self):
        svm = SVM(1)
        x, y = TestUtils.load_perceptron_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.7)
        svm.train(x_train, y_train)

        score = TestUtils.rate_classifier(svm, x_test, y_test)
        print("test_SVM", score)
        self.assertTrue(score > 0.95)


if __name__ == '__main__':
    unittest.main()
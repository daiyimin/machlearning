from StatisticLearning import CartTree
from StatisticLearning import DecisionTreeClassifier
from Tests import TestUtils
import unittest
import copy

class TestDecisionTree(unittest.TestCase):
    # def test_cart_tree(self):
    #     x, y = TestUtils.load_csv("TestData/DecisionTreeData.csv")
    #     classifier = DecisionTreeClassifier()
    #     classifier.fit(x, y)
    #     i = 0

    def test_DecisionTreeClassifier_with_car(self):
        # http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
        x, y = TestUtils.load_csv("TestData/car.data")
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.9)
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        score = TestUtils.rate_batch_classifier(classifier, x_test, y_test)
        print("test_DecisionTreeClassifier_with_car", score)
        self.assertTrue(score > 0.80)

    def test_DecisionTreeClassifier_with_iris(self):
        x, y = TestUtils.load_iris_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.90)

        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        score = TestUtils.rate_batch_classifier(classifier, x_test, y_test)
        print("test_DecisionTreeClassifier_with_iris", score)
        self.assertTrue(score > 0.80)


if __name__ == '__main__':
    unittest.main()
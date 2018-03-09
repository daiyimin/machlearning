from StatisticLearning import CartTree
from StatisticLearning import DecisionTreeClassifier
from StatisticLearning import Utils
from Tests import TestUtils
import unittest
import copy

class TestDecisionTree(unittest.TestCase):
    # def test_cart_tree(self):
    #     x, y = Utils.load_csv("TestData/DecisionTreeData.csv")
    #     classifier = DecisionTreeClassifier()
    #     classifier.fit(x, y)
    #     i = 0

    def test_DecisionTreeClassifier(self):
        x, y = Utils.load_csv("TestData/car.data")
        x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.9)
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        y_preds = classifier.predict(x_test)
        # score = classifier.rate_tree(y_preds, y_test)
        score = Utils.rate_batch_classifier(classifier, x_test, y_test)
        print(score)
        self.assertTrue(score > 0.80)

if __name__ == '__main__':
    unittest.main()
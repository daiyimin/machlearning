from StatisticLearning import KDTree
from StatisticLearning import KNeighboursClassifier
from StatisticLearning import Utils
from Tests import TestUtils
import unittest
import numpy as np

class TestKNNClassifier(unittest.TestCase):
    # def test_KDTree(self):
    #     x, y = TestUtils.load_knn_data()
    #     x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.90)
    #
    #     tree = KDTree()
    #     tree.build(x_train)
    #
    #     distance, neighbour = tree.searchKNeighbours(x_test, 3)

    def test_KNNClassifier(self):
        x, y = TestUtils.load_knn_data()
        x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.7)

        knn = KNeighboursClassifier(5)
        knn.fit(x_train, y_train)
        neighbour = knn.predict(x_test)

        score = Utils.rate_batch_classifier(knn, x_test, y_test)
        print(score)
        self.assertTrue(score > 0.90)

if __name__ == '__main__':
    unittest.main()
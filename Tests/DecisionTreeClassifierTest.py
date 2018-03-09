from StatisticLearning import CartTree
from StatisticLearning import Utils
from Tests import TestUtils
import unittest
import copy

class TestDecisionTree(unittest.TestCase):
    def test_cart_tree(self):
        tree = CartTree()
        x, y = Utils.load_pandas_data_from_csv("TestData/DecisionTreeData.csv")
        tree.build(x, y)
        tree.post_prune()
        i = 0

if __name__ == '__main__':
    unittest.main()
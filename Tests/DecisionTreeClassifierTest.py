from StatisticLearning import CartTree
from StatisticLearning import Utils
from Tests import TestUtils
import unittest

class TestDecisionTree(unittest.TestCase):
    def test_cart_tree(self):
        tree = CartTree()
        x, y = Utils.load_pandas_data_from_csv("TestData/DecisionTreeData.csv")
        tree.build(x, y)
        i = 1

if __name__ == '__main__':
    unittest.main()
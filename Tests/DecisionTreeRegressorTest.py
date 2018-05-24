from StatisticLearning import DecisionTreeRegressor
from StatisticLearning import GBDT
from Tests import TestUtils
import unittest
import numpy as np
import warnings

class TestDecisionTreeRegressor(unittest.TestCase):
    def test_CartRegressTree_with_iris(self):
        x, y = TestUtils.load_iris_data()
        x_train, y_train, x_test, y_test = TestUtils.model_selection(x, y, 0.9)

        regressor = DecisionTreeRegressor()
        regressor.fit(x_train, y_train)
        sqr_err = TestUtils.rate_batch_regressor(regressor, x_test, y_test, True)
        print("test_CartRegressTree_with_iris", sqr_err)

if __name__ == '__main__':
    unittest.main()
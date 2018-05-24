import numpy as np
from StatisticLearning.Models.GBDT import GBDT

class GBDTClassifier:
    def fit(self, x, y):
        self.boosting_tree = GBDT(5)
        self.boosting_tree.train(x, y)

    def predict(self, xs):
        categories = []
        # predict one by one in xs, and put in a list
        for x in xs:
            p_positive = self.boosting_tree.predict(x)
            category = 1 if p_positive >= 0.5 else -1
            categories.append(category)
        return np.array(categories)
import numpy as np
import pandas as pd
from StatisticLearning.Models.MaxEntroPyGIS import MaxEntroPyGIS

class MaxEntropyClassifier:

    # x, y are np.array type
    def fit(self, x, y):
        x_train = pd.DataFrame(x)
        y_train = pd.Series(y)

        self.max_ent = MaxEntroPyGIS()
        self.max_ent.train(x_train, y_train)

    # xs is np.array type
    def predict(self, xs, tree=None):
        categories = []
        # predict one by one in xs, and put in a list
        for x in xs:
            # Attention: MaxEntropy.predict requires Pandas.Series as parameter.
            prob = self.max_ent.fast_predict(pd.Series(x))
            max_Py = prob[1].max()
            category = prob[prob[1] == max_Py][0]
            categories.append(category)
        return np.array(categories)

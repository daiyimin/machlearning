import numpy as np
import pandas as pd
from StatisticLearning.Models.MaxEntropyGIS import MaxEntropyGIS
from StatisticLearning.Models.FastMaxEntropyGIS import FastMaxEntropyGIS

class MaxEntropyClassifier:
    fast_mode = True

    # x, y are np.array type
    def fit(self, x, y):
        x_train = pd.DataFrame(x)
        y_train = pd.Series(y)

        if self.fast_mode:
            self.max_ent = FastMaxEntropyGIS()
            self.max_ent.load_data(x_train, y_train)
            self.max_ent.train()
        else:
            self.max_ent = MaxEntropyGIS()
            self.max_ent.train(x_train, y_train)

    # xs is np.array type
    def predict(self, xs, tree=None):
        categories = []
        # predict one by one in xs, and put in a list
        for x in xs:
            # Attention: MaxEntropy.predict requires Pandas.Series as parameter.
            prob = self.max_ent.predict(pd.Series(x))
            max_Py = prob[0].max()
            category = prob[prob[0] == max_Py].loc[0, 1]
            categories.append(category)
        return np.array(categories)

from StatisticLearning.Models.Logistic import Logistic
import numpy as np

class LogisticClassifier:

    def fit(self, x_train, y_train):
        self.model = Logistic()
        self.model.train(x_train, y_train)

    def predict(self, x_test):
        prob = self.model.calculate_prob(x_test)

        categories = []
        for i in range(len(prob)):
            category = 1 if prob[i] > 0.5 else -1
            categories.append(category)

        return np.array(categories)
from StatisticLearning import Perceptron
from StatisticLearning import Utils
from Tests import TestUtils

perceptron = Perceptron(100)
x, y = TestUtils.load_perceptron_data()
x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.7)
perceptron.fit(x_train, y_train)

score = Utils.rate_classifier(perceptron, x_test, y_test)
print(score)

# Run the example in Chap02
# x, y = Utils.load_csv("TestData/PerceptionData.csv")
# x_train, y_train, x_test, y_test = Utils.model_selection(x, y, 0.7)
# perceptron.fit(x, y)
# score = Utils.rate_classifier(perceptron, x_test, y_test)
# print(score)

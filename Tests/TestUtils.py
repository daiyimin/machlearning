from sklearn import datasets

# load iris data for perceptron training
def load_perceptron_data():
    iris = datasets.load_iris()
    data = iris.data[iris.target < 2]
    target = iris.target[iris.target < 2] * 2 - 1
    return (data, target)

# load iris data for perceptron training
def load_knn_data():
    iris = datasets.load_iris()
    return (iris.data, iris.target)
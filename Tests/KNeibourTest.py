import numpy as np
from StatisticLearning import KDTree

data = np.array([[2,3], [5,4], [9,6], [4,7], [8,1], [7,2]])
y = np.array([1,2,1,1,3,4])
#data = np.array([[9,6,3], [4,7,4], [8,1,5], [7,2,2], [7,4,3], [3.2,5,1]])

tree = KDTree()
tree.build(data, y)

target = np.array([3,1])
neibour = tree.searchNearest(target)
neibour = tree.searchKNeibours(target, 2)
print(neibour)
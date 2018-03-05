import numpy as np
from collections import OrderedDict

'''
KNNSortedQueue is a utility class to save K nearest neighbour
'''
class KNNSortedQueue:
    # k: the number of nearest neighbor to be saved in the dictionary
    k = 1
    # knn: K nearest neighbour, in format of np.array([[index, distance],...])
    knn = None

    def __init__(self, k=1):
        self.k = k

    # push new neighbour(s)
    # nn: new neighbour(s) in format of np.array([[index, distance]...])
    #     Each line is a neighbour's data, including index and distance.
    def push(self, nn):
        if self.knn is None:
            self.knn = nn
        else:
            # concatenate new neighbours with knn
            self.knn = np.concatenate((self.knn, nn), axis=0)
            # sort knn by their distance to target (i.e. the last column of knn)
            # the distance is descendant order.
            self.knn = self.knn[np.lexsort(-self.knn.T)]
        if len(self.knn) > self.k:
            # if neighbours in knn is greater than k, only leave the k nearest neighbours
            self.knn = self.knn[-self.k:, :]

    # return greatest distance in knn
    def getMaxValue(self):
        return self.knn[0,1]

    # return number of neighbours
    def getSize(self):
        return len(self.knn)

    def getAll(self):
        return self.knn

# KDTree
class KDTree:
    def __init__(self, depth=0, parent=None):
        # depth is the depth of this tree in root tree. For root tree, its depth is 0.
        self.depth = depth
        # parent tree
        self.parent = parent
        # index of sharding feature
        self.sharding_feature = None
        self.median = None
        self.data = None
        self.children = dict()

    # x_train: each line is a data set, each column is one feature in data set
    def build(self, x_train):
        if self.depth == 0: # if this is root tree
            # append an indices column after feature columns
            # x_train[:, :-1] represents train data
            # x_train[:, -1:] represents indices of train data
            n_train = len(x_train)
            x_train_indices = np.arange(0, n_train)
            x_train_indices.shape = (n_train, 1)
            x_train = np.concatenate((x_train, x_train_indices), axis=1)

            # save the original train data in root node
            self.x_train = x_train

        # get the number of features (minus 1 to remove indices column)
        feature_num = x_train.shape[1] - 1
        # sharding_feature is the sharding feature
        self.sharding_feature = self.depth % feature_num

        # sort train data by the sharding feature
        sorted_idx = np.argsort(x_train[:, self.sharding_feature])
        sorted_x_train = x_train[sorted_idx]

        # get median of the sharding feature
        # because sharding feature is sorted, so the index of median is in the half middle
        median_idx = int(len(sorted_x_train)/2)
        self.median = sorted_x_train[median_idx, self.sharding_feature]

        # because there might be several value of sharding feature which equals to median
        # so try to get all data whose sharding feature value are equal to median
        median_idices = np.where(sorted_x_train[:, self.sharding_feature] == self.median)
        if np.size(median_idices) > 0:
            #  save those data to this tree node
            self.x = sorted_x_train[median_idices]

        # get indices of x train whose sharding feature value are less than median. They will be saved in left child.
        l_median_indices = np.where(sorted_x_train[:, self.sharding_feature] < self.median)
        # get indices of x train whose sharding feature value are bigger than median. They will be saved in right child.
        r_median_indices = np.where(sorted_x_train[:, self.sharding_feature] > self.median)

        # build child
        for indices, type in (l_median_indices,"left"), (r_median_indices, "right"):
            if np.size(indices) > 0:
                self.children[type] = KDTree(self.depth + 1, self)
                self.children[type].build(sorted_x_train[indices])

    #  search the nearest k neighbor of target
    #  target: target data to be searched
    #  k: number of neighbor to be returned
    #  queue: sorted queue of k nearest neighbours
    def search(self, target, k=1, queue=None):
        # calculate the distance between target and neighbours(x) on the super-plane of this tree nodeof this tree node
        # return their distance
        def distance(target, x):
            squared_difference = np.square(target - x)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            return np.array([distance]).T

        if self.depth == 0:
            # in the root tree, create queue as container of K nearest neighbours
            queue = KNNSortedQueue(k)

            train_data_num = len(self.x_train)
            if k >= train_data_num:
                # k is greater than train data number, no need to do search. Just add indices of train data to od_ball.
                x_distance = distance(target, self.x_train[:, :-1])
                # concatenate indices of x train and their distance to target
                nn = np.concatenate((self.x_train[:, -1:], x_distance), axis=1)
                # push knn into queue
                queue.push(nn)
                return queue

        # get distance of target to data on the super-plane of this tree node
        x_distance = distance(target, self.x[:, :-1])
        # concatenate indices of x train and their distance to target
        nn = np.concatenate((self.x[:, -1:], x_distance), axis=1)
        # push nn into queue
        queue.push(nn)

        #  depth first search in KD tree
        #  if target's sharding feature is less than self.median, search left child. Otherwise search right child.
        type = "left" if target[self.sharding_feature] <= self.median else "right"
        if type in self.children.keys():
            self.children[type].search(target, k, queue)

        # for a non-leaf node, we need to judge if their (not searched) children contains a nearest neighbor
        if len(self.children) > 0:
            # first, only when biggest super-ball intersects with split super-plane of current node
            # then the node's children will probably contains nearest neighbor
            #       D = distance between center of ball and super-plane = "np.abs(target[self.split] - self.median)"
            #       R = radius of ball = min_dist[0]
            #       if ( D < R ) then super-plane intersects with the ball
            # second, because we already search on child of current node during depth first search, now it's time
            # to try the other node
            if queue.getSize() < k or np.abs(target[self.sharding_feature] - self.median) < queue.getMaxValue():
                # when super-surface intersects with the ball, search the other child that didn't tried during depth first search
                type = "right" if target[self.sharding_feature] <= self.median else "left"
                if type in self.children.keys():
                    self.children[type].search(target, k, queue)
        return queue

    # search K nearest neighbours
    # targets: an array of targets to be serached
    # k: number of neighbours to be returned for each target
    # return an array of neighbours. Each line is indices of K neighbours of corresponding target
    def searchKNeighbours(self, targets, k=1):
        num_target = len(targets)
        distances = None
        neighbours = None
        for i in range(0, num_target):
            queue = self.search(targets[i], k)
            knn = queue.getAll()
            if distances is None:
                distances = knn[:, -1:].T
                neighbours = knn[:, :1].T
            else:
                distances = np.concatenate((distances, knn[:, -1:].T), axis=0)
                neighbours = np.concatenate((neighbours, knn[:, :1].T), axis=0)
        return distances, neighbours.astype(np.int32)
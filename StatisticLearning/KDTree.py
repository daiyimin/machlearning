import numpy as np
from collections import OrderedDict

"""
MinKOrderedDict is a utility class to save the nearest K neighbors during searching KDTree.
User can add newly found neighbor into MinKOrderedDict freely. When the number of neighbor is greater than K,
it will pop one farthest neighbor. The number of neighbor is always equal to (if not less than) K.
    key of the dict is distance between a neighbor and the target being searched. Each key represent a found super ball. 
        Radius of super ball equals to key.
    value of the dict is a list of neighbors whose is exactly distance "key" away from target. These neighbors are on
        the same super ball (belong to same key).
"""
class MinKOrderedDict(OrderedDict):
    # k: the number of nearest neighbor to be saved in the dictionary
    k = 1
    # capacity: the current capacity of dictionary, default value is 0
    capacity = 0

    def __init__(self, k=1):
        self.k = k
        OrderedDict.__init__(self)

    # return capacity
    def getCapacity(self):
        return self.capacity

    # The radius of the biggest super ball equals to MaxKey.
    # There are "capacity(<=K)" points in this biggest super ball.
    # return max key
    def getMaxKey(self):
        sorted_keys = sorted(self.keys())
        return sorted_keys[-1]

    # override the method of setting an item
    def __setitem__(self, key, value):
        containsKey = 1 if key in self else 0
        if containsKey:
            # when key already exists, it means we already find other neighbors who is distance="key" away from target
            # So, append the new value to the list of neighbor.
            item = OrderedDict.__getitem__(self, key)
            item.append(value)
            OrderedDict.__setitem__(self, key, item)
        else:
            # when key doesn't exist, it means this is the first neighbor who is distance "key" away from target
            # So add a new key and [value] pair.
            OrderedDict.__setitem__(self, key, [value])
        # increase capacity
        self.capacity += 1
        # check capacity value with k
        if self.capacity > self.k:
            # capacity is greater than k, we need to remove 1 farest neighbor
            # sort dict by key/distance
            sorted_keys = sorted(self.keys())
            # get the farest neighbor
            item = OrderedDict.__getitem__(self, sorted_keys[-1])
            if len(item) > 1:
                # if more than 1 neighbor share same distance, we just remove 1 of them from list
                item.pop()
            else:
                # otherwise, remove the item
                OrderedDict.__delitem__(self, sorted_keys[-1])
            # decrease capacity
            self.capacity -=1

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
    # y_train: each value is category of corresponding x_train data.
    def build(self, x_train, y_train):
        if self.depth == 0: # if this is root tree
            # save the original train data in root node
            self.x_train = x_train
            self.y_train = y_train

        # get the number of features
        feature_num = x_train.shape[1]
        # sharding_feature is the sharding feature
        self.sharding_feature = self.depth % feature_num

        # sort train data by the sharding feature
        sorted_idx = np.argsort(x_train[:, self.sharding_feature])
        sorted_x_train = x_train[sorted_idx]
        sorted_y_train = y_train[sorted_idx]

        # get median of the sharding feature
        # because sharding feature is sorted, so the index of median is in the half middle
        median_idx = int(len(sorted_x_train)/2)   # index of median, start from 0
        self.median = sorted_x_train[median_idx, self.sharding_feature] # value of median

        # because there might be several value of sharding feature which equals to median
        # so try to get all data whose sharding feature value are equal to median
        median_idices = np.where(sorted_x_train[:, self.sharding_feature] == self.median)
        if np.size(median_idices) > 0:
            #  save those data to this tree node
            self.x = sorted_x_train[median_idices]
            self.y = sorted_y_train[median_idices]

        # get indices of x train whose sharding feature value are less than median. They will be saved in left child.
        l_median_indices = np.where(sorted_x_train[:, self.sharding_feature] < self.median)
        # get indices of x train whose sharding feature value are bigger than median. They will be saved in right child.
        r_median_indices = np.where(sorted_x_train[:, self.sharding_feature] > self.median)

        # build child
        for indices, type in (l_median_indices,"left"), (r_median_indices, "right"):
            if np.size(indices) > 0:
                self.children[type] = KDTree(self.depth + 1, self)
                self.children[type].build(sorted_x_train[indices], sorted_y_train[indices])

    #  search the nearest neighbour(x) to the target
    #  target: target data to be searched
    #  min_distance: minimun distance
    #  nearest_x: nearest x
    #  y: corresponding y of nearest x
    #  return: min distance, nearest neighbour(x) to the target and its corresponding y
    def searchNearest(self, target, min_distance=-1, nearest_x=None, y=None):
        # calculate the distance between target and neighbours(x) on the super-plane of this tree node
        # return tuple of min distance, nearest neighbour(x) index, i.e. (min distance, nearest x index)
        def distance(target, x):
            squared_difference = np.square(target - x)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            min_distance = distance.min()
            nearest_x_idx = np.where(distance == min_distance)
            return min_distance, nearest_x_idx

        # get nearest neighbour(x) on the super-plane of this tree node
        local_min_distance, nearest_x_idx = distance(target, self.x)
        # if distance of nearest neighbour of this tree node is less than min distance, update min distance
        if min_distance < 0 or min_distance > local_min_distance:
            min_distance, nearest_x, y = local_min_distance, self.x[nearest_x_idx][0], self.y[nearest_x_idx][0]

        #  depth first search in KD tree
        #  if target's sharding feature is less than self.median, search left child. Otherwise search right child.
        type = "left" if target[self.sharding_feature] <= self.median else "right"
        if type in self.children.keys():
            min_distance, nearest_x, y = self.children[type].searchNearest(target, min_distance, nearest_x, y)

        # for a non-leaf node, we need to judge if their (not searched) children contains a nearest neighbor
        if len(self.children) > 0:
            # first, only when the super-ball intersects with split super-surface of current node
            # then the node's children will probably contains nearest neighbor
            #       D = distance between center of ball and super-surface = "np.abs(target[self.split] - self.median)"
            #       R = radius of ball = min_dist[0]
            #       if ( D < R ) then super-surface intersects with the ball
            # second, because we already search on child of current node during depth first search, now it's time
            # to try the other node
            if np.abs(target[self.sharding_feature] - self.median) < min_distance:
                # when super-surface intersects with the ball, search the other child that isn't tried during depth first search
                type = "right" if target[self.sharding_feature] <= self.median else "left"
                if type in self.children.keys():
                    min_distance, nearest_x, y = self.children[type].searchNearest(target, min_distance, nearest_x, y)
        return min_distance, nearest_x, y

    #  search the nearest k neighbor of target
    #  target: target data to be searched, type is numpy.array
    #  k: number of neighbor to be returned
    #  od_ball: dict of super balls, ordered by the key(=radius)
    def search(self, target, k=1, od_ball=None):
        # calculate the distance between target and neighbours(x) on the super-plane of this tree nodeof this tree node
        # return their distance
        def distance(target, x):
            squared_difference = np.square(target - x)
            squared_difference_sum = np.sum(squared_difference, axis=1)
            distance = np.sqrt(squared_difference_sum)
            return distance

        if self.depth == 0:
            # in the root tree, create an od_ball as container of K nearest neighbours
            od_ball = MinKOrderedDict(k)

            train_data_num = len(self.x_train)  # number of train data in KD tree
            if k >= train_data_num:
                # add all train data into od_ball
                # radius doesn't has meaning in this case, set them to range(0,train_data_num)
                for i in range(0, train_data_num):
                    od_ball[i] = self.x_train[i], self.y_train[i]
                return od_ball

        # get distance of data on the super-plane of this tree node
        # the distance is radius of super balls
        node_radius = distance(target, self.x)
        for i in range(0, len(self.x)):
            od_ball[node_radius[i]] = self.x[i], self.y[i]

        #  depth first search in KD tree
        #  if target's sharding feature is less than self.median, search left child. Otherwise search right child.
        type = "left" if target[self.sharding_feature] <= self.median else "right"
        if type in self.children.keys():
            self.children[type].search(target, k, od_ball)

        # for a non-leaf node, we need to judge if their (not searched) children contains a nearest neighbor
        if len(self.children) > 0:
            # first, only when biggest super-ball intersects with split super-plane of current node
            # then the node's children will probably contains nearest neighbor
            #       D = distance between center of ball and super-plane = "np.abs(target[self.split] - self.median)"
            #       R = radius of ball = min_dist[0]
            #       if ( D < R ) then super-plane intersects with the ball
            # second, because we already search on child of current node during depth first search, now it's time
            # to try the other node
            if od_ball.capacity < k or np.abs(target[self.sharding_feature] - self.median) < od_ball.getMaxKey():
                # when super-surface intersects with the ball, search the other child that didn't tried during depth first search
                type = "right" if target[self.sharding_feature] <= self.median else "left"
                if type in self.children.keys():
                    self.children[type].search(target, k, od_ball)
        return od_ball

    def searchKNeibours(self, target, k=1):
        od_ball = self.search(target, k)
        x_ret = None
        y_ret = None
        for key in od_ball.keys():
            data_list = od_ball.get(key)
            for x, y in data_list:
                x_np = np.array([x])
                y_np = np.array([y])
                if x_ret is None:
                    x_ret = x_np
                    y_ret = y_np
                else:
                    x_ret = np.concatenate((x_ret, x_np), axis=0)
                    y_ret = np.concatenate((y_ret, y_np))
        return x_ret, y_ret

    def traverse(self):
        """
        :return:  KD Tree in dictionary format
        """
        if len(self.children) > 0:
            tree = dict()
            sub_tree = dict()
            my_median = str(self.median)

            for child in self.children:
                op = "<" if child == "left" else ">"
                sub_tree[op + my_median] = self.children[child].traverse()

            tree_tag = "dim=" + str(self.sharding_feature) + ", data num =" + str(len(self.data))
            tree[tree_tag] = sub_tree
            return tree
        else:
            leaf_tag = "data num ="  + str(len(self.data))
            return leaf_tag

    # def draw(self):
    #     """
    #     draw KD Tree
    #     """
    #     tree_in_dict = self.traverse()
    #     plt.createPlot(tree_in_dict)

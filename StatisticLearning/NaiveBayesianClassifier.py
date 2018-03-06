class NaiveBayesianClassifier:

    def fit(self, train_x, train_y):
        def countUniqueVal(uniqueVal, data):
            unique_cnt = None
            for val in uniqueVal:
                cnt = np.array( [len(data[data==val])])
                if unique_cnt is None:
                    unique_cnt = cnt
                else:
                    unique_cnt = np.concatenate((unique_cnt, cnt))
            return unique_cnt

        unique_y_cnt = None
        # unique values of train_y
        unique_y = np.unique(train_y)
        # y_dist is the distribute of unique y in train_y
        for i in unique_y:
            cnt = np.array( [len(train_y[train_y==i])])
            if unique_y_cnt is None:
                unique_y_cnt = cnt
            else:
                unique_y_cnt = np.concatenate((unique_y_cnt, cnt))

        feature_num = train_x.shape[1]
        # probability of unique values per feature and y
        unique_x_prob_per_y_and_feature[i] = {}
        for i in range(0, feature_num):
            unique_x_of_feature = np.unique(train_x[:, i:i+1])
            # for this feature, calculate probability of unique x values per y
            x_prob_per_y = None
            for y in unique_y:
                x_slice_by_y = train_x[train_y==y, i:i+1]
                x_prob = countUniqueVal(unique_x_of_feature, x_slice_by_y) \
                         / unique_y_cnt[unique_y==y]
                if x_prob_per_y is None:
                    x_prob_per_y = np.array([x_prob])
                else:
                    x_prob_per_y = np.concatenate((x_prob_per_y, np.array([x_prob])), axis=0)
            unique_x_prob_per_y_and_feature[i] = x_prob_per_y
import numpy as np

class NaiveBayesianClassifier:

    def fit(self, train_x, train_y):
        # count the unique values in the input data
        # uniqueVal: unique values, for example np.array([val1, val2, ....])
        # data: input data
        # return
        #   count of unique values in input data, for example np.array([cnt1, cnt2,....])
        def countUniqueVal(uniqueVal, data):
            unique_cnt = None
            for val in uniqueVal:
                cnt = np.array( [len(data[data==val])])
                if unique_cnt is None:
                    unique_cnt = cnt
                else:
                    unique_cnt = np.concatenate((unique_cnt, cnt))
            return unique_cnt

        # unique values of train_y
        self.unique_y = np.unique(train_y)
        self.unique_y_cnt = countUniqueVal(self.unique_y, train_y)

        # number of feature in train x
        self.feature_num = train_x.shape[1]
        # map of probability of x
        # key is the index of feature, j = 0,1,2....
        # value is prob_xj_under_cond_y, see explanation of prob_xj_under_cond_y
        self.prob_x = {}
        for j in range(0, self.feature_num):
            # unique xj values (=ajl, l=1,2,...) in train data
            unique_xj = np.unique(train_x[:, j:j+1])
            # probability of each unique xj under condition of y==ck
            # line 0 is all unique xj values (=ajl, l=1,2,...) in train data
            # line k is conditional probability of all unique xj values for y=ck
            # column l is the conditional probability of xj==ajl for all unique y(=ck, k=1,2,...)
            # example: np.array([[aj1, aj2,....], [P(aj1|c1), P(aj2|c1)....], [P(aj1|c2), P(aj2|c2)....]....])
            prob_xj_under_cond_y = np.array([unique_xj])
            # calculate probability of each unique xj under condition of y==ck
            for ck in self.unique_y:
                # get xj slice of y=ck
                xj_slice = train_x[train_y==ck, j:j+1]
                # probability of all unique xj values (=ajl, l=1,2,...) in slice.
                # example: np.array([P(aj1|c1), P(aj2|c1)....])
                prob_xj_in_slice = countUniqueVal(unique_xj, xj_slice) / len(xj_slice)
                # concatecate prob_xj_under_cond_y and prob_xj_in_slice
                prob_xj_under_cond_y = np.concatenate((prob_xj_under_cond_y, np.array([prob_xj_in_slice])), axis=0)
            # save the prob_xj_under_cond_y of feature j
            self.prob_x[j] = prob_xj_under_cond_y

    def predict(self, test_xs):
        flags = None
        for x in test_xs:
            # prob is the probability of all unique y(=ck, k=1,2,...) for this x.
            # example: np.array([prob(c1|X), prob(c2|X), ...,prob（ck|X),...])
            # prob（ck|X) = sigma(ck)/N * P(X1==x1| y==ck) * P(X2==x2| y==ck) * ....  See (4.5) on P48 of the book
            # set initial value of prob to np.array([sigma(c1), sigma(c2),...]. N is same for all ck, it can be ignored
            prob = self.unique_y_cnt
            for j in range(0, self.feature_num):
                # get unique xj values (=ajl, l=1,2,...) in train data
                # example: np.array([aj1, aj2,....])
                unique_xj = self.prob_x[j][0]
                if not x[j] in unique_xj:
                    # if the j-th feature of x doesn't exist in unique_xj of train data, it's not possible to
                    # predict the y with Bayesian classifier. Therefore, set all prob（y == ck|X) to zero. It means
                    # all ck are impossible.
                    prob = np.zeros(len(self.unique_y))
                    break
                else:
                    # get the index of x[j] in unqiue xj array
                    index = np.where(unique_xj == x[j])[0][0] #[0][0] to get index value from array(tuple(index))
                    # use index to get the probability of xj==x[j] for all unique y(=ck, k=1,2,...).
                    # remove line 0 which is unique x values instead of probability.
                    # example: prob_Xj = np.array([P(Xj|c1), P(Xj|c2),..., P(Xj|ck)....])
                    prob_Xj = self.prob_x[j][1:, index]
                    # multiple all prob together
                    prob = prob * prob_Xj

            if np.max(prob) == 0:
                # if prob is all 0, it means all ck are not possible. Set flag to -np.inf which is not a valid value.
                flag = np.array([-np.inf])
            else:
                # find the index of most probable y in prob
                # remember that, prob = np.array([prob(c1|X), prob(c2|X), ...,prob（ck|X),...])
                most_probable_y_idx = np.argsort(prob)[-1]
                # get value of most probable y
                most_probable_y = self.unique_y[most_probable_y_idx]
                # use it as prediction flag of this x
                flag = np.array([most_probable_y])

            # concatenate flags of all targets into one big np.array
            if flags is None:
                flags = flag
            else:
                flags = np.concatenate((flags, flag), axis=0)

        return flags
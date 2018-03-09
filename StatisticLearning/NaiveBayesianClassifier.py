import numpy as np

class NaiveBayesianClassifier:
    # if laplace is not 0, then use Bayesian Estimation to calculate classifier parameters
    # if laplace is 0,use Most Likelihood Estimation to calculate classifier parameters
    laplace = 0
    # unique values of train_y
    unique_y = None
    # count of unique values of train_y
    unique_y_cnt = None
    # number of features in train_x
    feature_num = None
    # map of probability of x
    # prob_x[j], it's a np.array of shape(K+1,Sj)
    # in line 0, it is all unique xj(=ajl, l=1,2,..., Sj) in train data
    # in line k, it's the distribution of conditional probability of Xj|Y=ck, Xj=ajl, l=1,2,..., Sj
    prob_x = None

    def __init__(self, laplace=0):
        self.laplace = laplace

    def fit(self, train_x, train_y):
        # count the unique values in the input data
        # uniqueVal: unique values, for example np.array([val1, val2, ....])
        # data: input data
        # return
        #   count of unique values in input data, for example np.array([cnt1, cnt2,....])
        def value_counts(uniqueVal, data):
            unique_cnt = []
            for val in uniqueVal:
                cnt = len(data[data==val])
                unique_cnt.append(cnt)
            return np.array(unique_cnt).astype(np.float64)

        # unique values of train_y
        self.unique_y = np.unique(train_y)
        # unique value of train_y
        self.unique_y_cnt = value_counts(self.unique_y, train_y)
        # adjust unique value of train_y with laplace coefficient
        self.unique_y_cnt += self.laplace

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
            # line 0 is all unique xj(=ajl, l=1,2,...) in train data
            # line k is the distribution of conditional probability of Xj|Y=ck, Xj=ajl, l=1,2,..., Sj
            # column l-1 is the conditional probability of xj==ajl for all unique y(=ck, k=1,2,...)
            # example: np.array([[aj1, aj2,....], [P(aj1|c1), P(aj2|c1)....], [P(aj1|c2), P(aj2|c2)....]....])
            prob_xj_under_cond_y = np.array([unique_xj])
            # calculate probability of each unique xj under condition of y==ck
            for ck in self.unique_y:
                # get xj slice of y=ck
                xj_slice = train_x[train_y==ck, j:j+1]
                # adjust the slice length with laplace coefficient
                adjusted_slice_len = len(xj_slice) + len(unique_xj) * self.laplace
                # adjust the unique xj count in xj_slice with laplace coefficient
                adjusted_unique_xj_cnt = value_counts(unique_xj, xj_slice) + self.laplace
                # probability of all unique xj values (=ajl, l=1,2,...) in slice.
                # example: np.array([P(aj1|c1), P(aj2|c1)....])
                prob_xj_in_slice = adjusted_unique_xj_cnt / adjusted_slice_len
                # concatecate prob_xj_under_cond_y and prob_xj_in_slice
                prob_xj_under_cond_y = np.concatenate((prob_xj_under_cond_y, np.array([prob_xj_in_slice])), axis=0)
            # save the prob_xj_under_cond_y of feature j
            self.prob_x[j] = prob_xj_under_cond_y

    def predict(self, xs):
        flags = []
        for x in xs:
            # prob is the probability of all unique y(=ck, k=1,2,...) for this x.
            # example: np.array([prob(c1|X), prob(c2|X), ...,prob（ck|X),...])
            # prob（ck|X) = sigma(ck)/N * P(X1==x1| y==ck) * P(X2==x2| y==ck) * ....  See (4.5) on P48 of the book
            # set initial value of prob to np.array([sigma(c1), sigma(c2),...].
            # N + laplace * K is same for all ck, it can be ignored
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
                    prob = prob * prob_Xj.astype(np.float64)

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

            flags.append(flag)

        return np.array(flags)
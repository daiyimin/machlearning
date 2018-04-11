import numpy as np

class SVM:
    num_iterations = None
    precision = None
    C = None

    # SVM model data
    alpha = None
    b = None

    def __init__(self, C, num_iterations=2000, precision = 0.01):
        self.C = C
        self.num_iterations = num_iterations
        self.precision = precision

    # calculate all K(xi,xj), return them in a matrix
    def calculate_K(self, x):
        # K(xi,xj) = np.dot(xi,xj)
        K = np.dot(x, x.T)
        return K

    def calculate_LH(self, y1, y2, alpha1_old, alpha2_old):
        if y1 == y2:
            L = max(0, alpha2_old + alpha1_old - self.C)
            H = min(self.C, alpha2_old + alpha1_old)
        else:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, alpha2_old - alpha1_old + self.C)
        return L, H

    def trunc_alpha2(self, alpha2, L, H):
        alpha2 = H if alpha2 > H else alpha2
        alpha2 = L if alpha2 < L else alpha2
        return alpha2

    def iterate(self, alpha, b, i1, Y, K, E):
        # select alpha_2 which maximize abs(E1 - E2)
        E1 = E[i1]
        delta_E = abs(E - E1)
        selected_i2 = np.argsort(delta_E)[-1]

        y1 = Y[i1]
        y2 = Y[selected_i2]
        alpha1_old = alpha[i1]
        alpha2_old = alpha[selected_i2]
        L, H = self.calculate_LH(y1, y2, alpha1_old, alpha2_old)
        if L >= H:
            print("L>=H"); return None

        # calculate eta in 7.107
        K11 = K[i1, i1]
        K22 = K[selected_i2, selected_i2]
        K12 = K[i1, selected_i2]
        eta = K11 + K22 - 2 * K12
        if eta < 0: print("eta<0"); return None

        # calculate alpha2_new_unc in 7.106
        E2 = E[selected_i2]
        alpha2_new_unc = alpha2_old + y2 * (E1 - E2) / eta
        # calculate alpha2_new in 7.108
        alpha2_new = self.trunc_alpha2(alpha2_new_unc, L, H)
        # calculate alpha1_new in 7.109
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

        # update alpha
        alpha[i1] = alpha1_new
        alpha[selected_i2] = alpha2_new
        # calculate b_new
        b1 = -E1 - y1 * K11 * (alpha1_new - alpha1_old) - y2 * K12 * (alpha2_new - alpha2_old) + b
        b2 = -E2 - y1 * K12 * (alpha1_new - alpha1_old) - y2 * K22 * (alpha2_new - alpha2_old) + b
        b_new = (b1 + b2) / 2

        return b_new

    def train(self, X_train, Y_train):
        N = len(X_train)
        K = self.calculate_K(X_train)

        alpha = np.zeros(N)
        b = 0
        for iter in range(self.num_iterations):
            # calculate gxi in 7.104, gxi = gx[i]
            gx = np.dot(alpha * Y_train, K) + b
            # calculate Ei in 7.105, Ei = E[i]
            E = gx - Y_train

            # select an alpha_1 that violate KKT condition: when 0 < alpha_i < C, yi*gxi == 1
            selected_i1 = None
            for i in range(N):
                if alpha[i] > 0 and alpha[i] < self.C and abs(E[i]) > self.precision:
                    # check KKT condition: when 0 < alpha_i < C, yi*gxi == 1 <==> Ei == 0
                    # if abs(Ei) > precision, then KKT condition is violated, select i as i1, i.e. alpha[i] as alpha1
                    selected_i1 = i
                    # iterate to get alpha1_new, alpha2_new and b_new
                    b_new = self.iterate(alpha, b, selected_i1, Y_train, K, E)
                    if b_new is None:
                        # if iterate fails, continue to try next i
                        selected_i1 = None; continue
                    else:
                        # update b with b_new
                        b = b_new
                        break

            if selected_i1 is None:
                # select an alpha_1 that violate KKT condition: when alpha_i == 0, yi*gxi >= 1 or when alpha_i == C, yi*gxi <= 1
                for i in range(N):
                    if (alpha[i] == 0 and Y_train[i]*gx[i] < 1 - self.precision) or \
                        (alpha[i] == self.C and Y_train[i]*gx[i] > 1 + self.precision):
                        # check KKT condition: when alpha_i == 0, yi*gxi >= 1，if yi*gxi  < 1 - precision, then KKT condition is violated
                        # check KKT condition: when alpha_i == C, yi*gxi <= 1，if yi*gxi  > 1 + precision, then KKT condition is violated
                        selected_i1 = i
                        # iterate to get alpha1_new, alpha2_new and b_new
                        b_new = self.iterate(alpha, b, selected_i1, Y_train, K, E)
                        if b_new is None:
                            # if iterate fails, continue to try next i
                            selected_i1 = None; continue
                        else:
                            # update b with b_new
                            b = b_new
                            break

            if selected_i1 is None:
                # when no KKT violation found, stop iteration
                break

        self.alpha = alpha
        self.b = b
        self.x = X_train
        self.y = Y_train

    def predict(self, x):
        K = np.dot(self.x, x)
        # calculate f in 7.94
        f = np.dot(self.alpha*self.y, K) + self.b
        return 1 if f >=0 else -1
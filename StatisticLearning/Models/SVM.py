import numpy as np

class SVM:
    num_iterations = None
    precision = None
    C = None

    # SVM model data
    alpha = None
    b = None
    x = None
    y = None

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

    def iterate(self, i1, K, E):
        # select alpha_2 which maximize abs(E1 - E2)
        # BTW, because abs(E1 - E1) = 0 <= abs(E1 - E2), so i2 will never be same as i1.
        E1 = E[i1]
        delta_E = abs(E - E1)
        selected_i2 = np.argsort(delta_E)[-1]

        y1 = self.y[i1]
        y2 = self.y[selected_i2]
        alpha1_old = self.alpha[i1]
        alpha2_old = self.alpha[selected_i2]
        L, H = self.calculate_LH(y1, y2, alpha1_old, alpha2_old)
        if L >= H:
            print("L>=H"); return False

        # calculate eta in 7.107
        K11 = K[i1, i1]
        K22 = K[selected_i2, selected_i2]
        K12 = K[i1, selected_i2]
        eta = K11 + K22 - 2 * K12
        if eta < 0: print("eta<0"); return False

        # calculate alpha2_new_unc in 7.106
        E2 = E[selected_i2]
        alpha2_new_unc = alpha2_old + y2 * (E1 - E2) / eta
        # calculate alpha2_new in 7.108
        alpha2_new = self.trunc_alpha2(alpha2_new_unc, L, H)
        # calculate alpha1_new in 7.109. 7.109 can ensure that KKT condition 7.23 is always satisfied. Reason:
        # al2_n*y2 + al1_n*y1 = al2_o*y2 + al1_o*y1
        # al1_n*y1 = al1_o*y1 + y2(al2_o - al2_n)
        # al1_n = al1_o + y1*y2*（al2_o - al2_n)
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)

        # update alpha
        self.alpha[i1] = alpha1_new
        self.alpha[selected_i2] = alpha2_new
        # calculate b_new
        b1 = -E1 - y1 * K11 * (alpha1_new - alpha1_old) - y2 * K12 * (alpha2_new - alpha2_old) + self.b
        b2 = -E2 - y1 * K12 * (alpha1_new - alpha1_old) - y2 * K22 * (alpha2_new - alpha2_old) + self.b
        b_new = (b1 + b2) / 2

        # update b
        self.b = b_new

        return True

    def train(self, X_train, Y_train):
        N = len(X_train)
        K = self.calculate_K(X_train)

        # initialize model data
        self.x = X_train
        self.y = Y_train
        self.alpha = np.zeros(N)
        self.b = 0
        for iter in range(self.num_iterations): # outer loop
            # calculate gxi in 7.104, gxi = gx[i]
            gx = np.dot(self.alpha * self.y, K) + self.b
            # calculate Ei in 7.105, Ei = E[i]
            E = gx - self.y

            # select an alpha_1 that violate KKT condition: when 0 < alpha_i < C, yi*gxi == 1 <==> Ei == 0
            selected_i1 = None
            for i in range(N): # inner loop
                if self.alpha[i] > 0 and self.alpha[i] < self.C and abs(E[i]) > self.precision:
                    # if abs(Ei) > precision, then KKT condition is violated, select i as i1, i.e. alpha[i] as alpha1
                    selected_i1 = i
                    # iterate to get alpha1_new, alpha2_new and b_new
                    success = self.iterate(selected_i1, K, E)
                    if success:
                        break # stop inner loop if successful
                    else:
                        # if iterate fails, continue inner loop to try next i
                        selected_i1 = None; continue

            if selected_i1 is None:
                # select an alpha_1 that violate KKT condition: when alpha_i == 0, yi*gxi >= 1 or when alpha_i == C, yi*gxi <= 1
                for i in range(N): # inner loop
                    if (self.alpha[i] == 0 and self.y[i]*gx[i] < 1 - self.precision) or \
                        (self.alpha[i] == self.C and self.y[i]*gx[i] > 1 + self.precision):
                        # check KKT condition: when alpha_i == 0, yi*gxi >= 1，if yi*gxi  < 1 - precision, then KKT condition is violated
                        # check KKT condition: when alpha_i == C, yi*gxi <= 1，if yi*gxi  > 1 + precision, then KKT condition is violated
                        selected_i1 = i
                        # iterate to get alpha1_new, alpha2_new and b_new
                        success = self.iterate(selected_i1, K, E)
                        if success:
                            break  # stop inner loop
                        else:
                            # if iterate fails, continue inner loop to try next i
                            selected_i1 = None; continue

            if selected_i1 is None:
                # when no KKT violation found, stop iteration
                break

    def predict(self, x):
        K = np.dot(self.x, x)
        # calculate f in 7.94
        f = np.dot(self.alpha*self.y, K) + self.b
        return 1 if f >=0 else -1
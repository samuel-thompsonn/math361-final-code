import numpy as np


def dx(t, x):
    return


def dv(x, N, H, i):
    # first term: (alpha - beta*|v_i|^2)*v_i, where v_i is the velocity of the bird
    v_i = x[(4*i)+2:(4*i)+4]
    x_i = x[4*i:(4*i)+2]
    curr_sum = np.array([0, 0])
    for j in range(N):
        x_j = x[j*4:(j*4)+2]
        v_j = x[(j*4)+2:(j*4)+4]
        curr_sum = curr_sum + H(np.linalg.norm(x_i - x_j)) * (v_j - v_i)
    return curr_sum * (1 / N)


class CuckerSmaleModel:
    def __init__(self, K, sigma, gamma, N):
        self.K = K
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, t, y):
        y_prime = np.empty([4 * self.N])
        for i in range(0, self.N):
            y_prime[4 * i:(4 * i) + 2] = y[(4 * i) + 2:(4 * i) + 4]
            y_prime[(4 * i) + 2:(4 * i) + 4] = dv(y, self.N, self.communication_rate, i)
        return y_prime

    def communication_rate(self, r):
        return self.K / ((self.sigma**2) + (r**2))**self.gamma

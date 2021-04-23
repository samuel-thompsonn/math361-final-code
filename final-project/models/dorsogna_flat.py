import numpy as np


def dx(t, x):
    return


# https://tinyurl.com/swarm361
def dv(x, alpha, beta, N, U_grad, i):
    # first term: (alpha - beta*|v_i|^2)*v_i, where v_i is the velocity of the bird
    v = x[(4*i)+2:(4*i)+4]

    first_term = (alpha - beta*(np.linalg.norm(v))**2) * v
    second_term = calc_second_term(x, U_grad, i, N)
    return first_term - second_term


def calc_second_term(x, U_grad, i, N):
    curr_sum = np.array([0, 0])
    for j in range(N):
        if i == j:
            continue
        curr_sum = curr_sum + U_grad(x[i*4:(i*4)+2], x[j*4:(j*4)+2])
    return curr_sum * (1 / N)


class DorsognaModel:
    def __init__(self, alpha, beta, C_a, ell_a, C_r, ell_r, N):
        self.alpha = alpha
        self.beta = beta
        self.C_a = C_a
        self.ell_a = ell_a
        self.C_r = C_r
        self.ell_r = ell_r
        self.N = N

    def f(self, t, y):
        y_prime = np.empty([4 * self.N])
        for i in range(0, self.N):
            y_prime[4 * i:(4 * i) + 2] = y[(4 * i) + 2:(4 * i) + 4]
            y_prime[(4 * i) + 2:(4 * i) + 4] = dv(y, self.alpha, self.beta, self.N, self.morse_potential_gradient, i)
        return y_prime

    def morse_potential_gradient(self, x, y):
        # C_a = 100  # attraction strength
        # C_r = 200  # repulsion (collision avoidance) strength
        # el_a = 10  # attraction distance
        # el_r = 0.05  # repulsion distance

        r = np.linalg.norm(x - y)
        first_term = (-r / self.ell_r) * (self.C_r * np.exp(-r / self.ell_r))
        second_term = (-r / self.ell_a) * (-self.C_a * np.exp(-r / self.ell_a))
        return ((x - y) / r) * (first_term + second_term)
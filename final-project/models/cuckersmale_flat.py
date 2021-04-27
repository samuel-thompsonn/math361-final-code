import numpy as np


def dv(x, N, H, i):
    # Returns the change in velocity given that x has the correct flat layout
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
        """
        Simulates 2D flock-like behavior as described in section 2.3 of https://tinyurl.com/swarm361.
        Includes alignment of agents' velocities and the range and effectiveness of communication between agents.
        :param K: Magnitude of communication rate
        :param sigma: Inverse of maximum communication rate
        :param gamma: Order of magnitude of decay in communication rate with increasing distance
        :param N: The number of agents
        """
        self.K = K
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def f(self, t, y):
        """
        Calculates the derivative of velocity and position for each agent
        :param t: Current time
        :param y: Current positions and velocities of agents, where agent i has position y[4*i:(4*i)+1] and
                    velocity y[(4*i)+2:(4*i)+4]
        :return: The derivative y'(y,t) according to the Cucker-Smale model with specified parameters
        """
        y_prime = np.empty([4 * self.N])
        for i in range(0, self.N):
            y_prime[4 * i:(4 * i) + 2] = y[(4 * i) + 2:(4 * i) + 4]
            y_prime[(4 * i) + 2:(4 * i) + 4] = dv(y, self.N, self.communication_rate, i)
        return y_prime

    def communication_rate(self, r):
        # Returns H(r) as described in the papers, with instance variable parameters
        return self.K / ((self.sigma**2) + (r**2))**self.gamma

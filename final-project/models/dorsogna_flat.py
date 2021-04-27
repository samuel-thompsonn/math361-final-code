import numpy as np


# Described in section 2.2 of https://tinyurl.com/swarm361
def dv(x, alpha, beta, N, U_grad, i):
    # Returns the change in velocity described by the D'Orsogna et al. model
    # first term: (alpha - beta*|v_i|^2)*v_i, where v_i is the velocity of the bird
    v = x[(4*i)+2:(4*i)+4]

    first_term = (alpha - beta*(np.linalg.norm(v))**2) * v
    second_term = calc_second_term(x, U_grad, i, N)
    return first_term - second_term


def calc_second_term(x, U_grad, i, N):
    # Returns the second term of dv/dt, which depends on the potential function and distance from all other agents
    curr_sum = np.array([0, 0])
    for j in range(N):
        if i == j:
            continue
        curr_sum = curr_sum + U_grad(x[i*4:(i*4)+2], x[j*4:(j*4)+2])
    return curr_sum * (1 / N)


class DorsognaModel:
    def __init__(self, alpha, beta, C_a, ell_a, C_r, ell_r, N):
        """
        Simulates 2D flock-like behavior as described in section 2.2 of https://tinyurl.com/swarm361.
        Includes attraction between agents, repulsion between agents, and air resistance.
        :param alpha: The magnitude of agents' self-propulsion in the direction they're already facing
        :param beta: The magnitude of air resistance
        :param C_a: Strength of attraction between agents
        :param ell_a: Typical distance at which agents are attracted
        :param C_r: Strength of repulsion between agents
        :param ell_r: Typical distance at which agents are repulsed from each other
        :param N: The number of agents
        """
        self.alpha = alpha
        self.beta = beta
        self.C_a = C_a
        self.ell_a = ell_a
        self.C_r = C_r
        self.ell_r = ell_r
        self.N = N

    def f(self, t, y):
        """
        Calculates the derivative of velocity and position for each agent
        :param t: Current time
        :param y: Current positions and velocities of agents, where agent i has position y[4*i:(4*i)+1] and
                    velocity y[(4*i)+2:(4*i)+4]
        :return: The derivative y'(y,t) according to the D'Orsogna et al. model with specified parameters
        """
        y_prime = np.empty([4 * self.N])
        for i in range(0, self.N):
            y_prime[4 * i:(4 * i) + 2] = y[(4 * i) + 2:(4 * i) + 4]
            y_prime[(4 * i) + 2:(4 * i) + 4] = dv(y, self.alpha, self.beta, self.N, self.morse_potential_gradient, i)
        return y_prime

    def morse_potential_gradient(self, x, y):
        # Returns the net attractive force between agents at positions x and y according to Morse potential
        r = np.linalg.norm(x - y)
        first_term = (-1 / self.ell_r) * (self.C_r * np.exp(-r / self.ell_r))
        second_term = (-1 / self.ell_a) * (-self.C_a * np.exp(-r / self.ell_a))
        return ((x - y) / r) * (first_term + second_term)

import numpy as np
import read_fraction as frac


def get_line(file):
    # Retrieves the next line from the file that isn't blank or a comment.
    s = file.readline()
    while s == '\n' or s[0] == '#':
        s = file.readline()
    return s


def line_as_floats(file, length=0):
    # Attempts to retrieve the next line from the file as a list of whitespace-separated floats.
    # Pads the returned list to size 'length' if the line is too short.
    arr = []
    for term in get_line(file).split():
        arr.append(frac.parse_fraction(term))
    if len(arr) < length:
        arr += [0]*(length - len(arr))
    return np.array(arr)


def read_embedded_table(filepath):
    """
    Reads the file as a Butcher tableau for an embedded pair of RK methods.
    :param filepath: The relative location of the file containing the table.
    :return:
        c: Offsets of t for each k
        a: Lower triangular matrix of coefficients, where a[i][j] is multiplied by k_j when calculating y value
            for k_i
        b: Weight for each k when they are summed and added to y_n to calculate y_n+1
        s: number of k values
    """
    f = open(filepath, "r")
    s = int(get_line(f))
    c = line_as_floats(f)
    a = np.zeros([s, s-1])
    for i in range(1, s):
        a[i, :] = line_as_floats(f, s-1)
    b = np.empty([2, s])
    for i in range(2):
        b[i, :] = line_as_floats(f, s)
    f.close()
    return c, a, b, s


class AdaptiveSolver:
    def __init__(self, filepath, epsilon):
        """
        Uses the specified Butcher tableau to approximate ODE values with adaptive time steps.
        :param filepath: The relative file location of the embedded RK pair Butcher tableau
        :param epsilon: The maximum error tolerance when calculating step size
        """
        self.epsilon = epsilon
        self.c, self.a, self.b, self.s = read_embedded_table(filepath)

    def calc_k(self, f, y_n, t_n, h):
        # Calculates the k-values shared between the RK methods and returns them as an array
        k = np.empty([self.s, y_n.size])
        for i in range(self.s):
            y_i = y_n.copy()
            for j in range(i):
                y_i += h * self.a[i, j] * k[j]
            t_i = t_n + h * self.c[i]
            k[i] = f(t_i, y_i)
        return k

    def approx_with_method(self, y_n, h, k, method_num):
        # Uses one of the two methods (the higher order one or the lower order one) to approximate y_n+1
        y_next = y_n.copy()
        for i in range(self.s):
            y_next += h * self.b[method_num, i] * k[i]
        return y_next

    def approx_step(self, f, t_n, y_n, h, epsilon):
        # Approximates one step from y_n to y_n+1 and recalculates step size
        # First, find the new h using the rk approximations and previous h
        k = self.calc_k(f, y_n, t_n, h)
        y_tilde = self.approx_with_method(y_n, h, k, 1)
        y_hat = self.approx_with_method(y_n, h, k, 0)
        trunc_error = np.linalg.norm(y_tilde - y_hat)
        h_new = 0.8 * h * (epsilon / trunc_error)**(1/5)
        h = h_new

        # Next, use h_new to come up with the actual approximation
        k = self.calc_k(f, y_n, t_n, h)
        y_hat = self.approx_with_method(y_n, h, k, 0)
        return h, y_hat

    def ode_approx(self, f, a, b, y0, h):
        """
        Approximates vector-valued function y(b) starting at y(a) = y0, with initial time step h.
        :param f: The derivative function for y and t
        :param a: The initial t-value
        :param b: The final t-value
        :param y0: The initial y-value y(a)
        :param h: The initial step size used to approximate the error that determines the size of the first time step
        :return:
            yvals: A numpy array of the value of y after each time step
            tvals: A numpy array of the value of t after each time step
        """
        t_n = a
        y_n = y0
        yvals = np.array([y0])
        tvals = [a]

        while t_n < b:
            h, y_n = self.approx_step(f, t_n, y_n, h, self.epsilon)
            t_n += h
            yvals = np.vstack((yvals, [y_n]))
            tvals.append(t_n)

        return yvals, tvals

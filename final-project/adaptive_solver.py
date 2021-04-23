import numpy as np
import read_fraction as frac

def get_line(file):
    s = file.readline()
    while s == '\n' or s[0] == '#':
        s = file.readline()
    return s


def line_as_floats(file, length=0):
    arr = []
    for term in get_line(file).split():
        arr.append(frac.parse_fraction(term))
    if len(arr) < length:
        arr += [0]*(length - len(arr))
    return np.array(arr)


def read_embedded_table(filepath):
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
        self.epsilon = epsilon
        self.c, self.a, self.b, self.s = read_embedded_table(filepath)

    def calc_k(self, f, y_n, t_n, h):
        k = np.empty([self.s, y_n.size])
        for i in range(self.s):
            y_i = y_n.copy()
            for j in range(i):
                y_i += h * self.a[i, j] * k[j]
            t_i = t_n + h * self.c[i]
            k[i] = f(t_i, y_i)
        return k

    def approx_with_method(self, y_n, h, k, method_num):
        y_next = y_n.copy()
        for i in range(self.s):
            y_next += h * self.b[method_num, i] * k[i]
        return y_next

    def approx_step(self, f, t_n, y_n, h, epsilon):
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

import numpy as np
import read_fraction as frac


class TableauSolver:
    def __init__(self, filepath):
        f = open(filepath, "r")
        self.tableau = []
        for line in f:
            numbers = [frac.parse_fraction(x) for x in line.split()]
            self.tableau.append(numbers)
        f.close()

    def ode_approx(self, f, a, b, y0, h):
        t_n = a
        y_n = y0
        yvals = np.array([y0])
        tvals = [a]

        while t_n < b:
            y_next = y_n.copy()
            k = np.empty([len(self.tableau) - 1, len(y0)])
            for i in range(len(self.tableau) - 1):
                y_start = y_n.copy()
                for j in range(1, i + 1):
                    y_start += h * self.tableau[i][j] * k[j - 1]
                k[i] = f(t_n + h * self.tableau[i][0], y_start)
                y_next += k[i] * self.tableau[len(self.tableau) - 1][i + 1] * h

            y_n = y_next
            t_n += h
            yvals = np.vstack((yvals, [y_next]))
            tvals.append(t_n)

        return yvals, tvals

import timeit

import tableau_solver as btab
import numpy as np
import matplotlib.pyplot as plt
import models.dorsogna_flat as model


def circle_func(t, y):
    """
    :param t: independent (time) variable
    :param y: 2d numpy array
    :return: tangential velocity for circular motion
    """
    A = np.array([[0, -1], [1, 0]])
    return A @ y


def init_flock(N, seed):
    generator = np.random.RandomState(seed)
    positions = np.empty([N, 4])
    for i in range(N):
        for j in range(2):
            positions[i][j] = generator.uniform(-1, 1)
    return positions.reshape(N*4)


if __name__ == '__main__':
    alpha = 1
    beta = 1
    C_a = 100
    ell_a = 10
    C_r = 200
    ell_r = 1
    N = 5
    rk_solver = btab.TableauSolver("tables/rk4_tableau.txt")
    euler_solver = btab.TableauSolver("tables/rk4_tableau.txt")
    y0 = init_flock(N, 0)

    flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, N)

    starttime = timeit.default_timer()
    xvals, tvals = rk_solver.ode_approx(flock.f, 0, 30, y0, 0.0125)
    endtime = timeit.default_timer()
    print("Runtime for RK4 approximation: {0}\n".format(endtime - starttime))
    points = xvals.T
    euler_x, euler_t = euler_solver.ode_approx(flock.f, 0, 40, y0, 0.0125)
    euler_points = euler_x.T
    fig = plt.figure(1)
    plt.title("RK4 Approximation")
    # plt.plot(euler_points[0], euler_points[1])
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.figure(2)
    plt.title("Euler Approximation")
    for i in range(N):
        plt.plot(euler_points[4 * i], euler_points[(4 * i) + 1])
    plt.show()


def test_circle():
    rk_solver = btab.TableauSolver("tables/rk4_tableau.txt")
    euler_solver = btab.TableauSolver("tables/rk4_tableau.txt")
    y0 = np.array([1., 0])

    xvals, tvals = rk_solver.ode_approx(circle_func, 0, 160, y0, 0.0125)
    euler_x, euler_t = euler_solver.ode_approx(circle_func, 0, 160, y0, 0.0125)
    points = xvals.T
    euler_points = euler_x.T
    fig = plt.figure()
    plt.plot(euler_points[0], euler_points[1])
    plt.plot(points[0], points[1])
    plt.show()

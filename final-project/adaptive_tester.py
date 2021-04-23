import adaptive_solver as adps
import numpy as np
import matplotlib.pyplot as plt
import models.dorsogna_flat as model
import timeit


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


def test_circle():
    epsilon = 10e-12
    rk_solver = adps.AdaptiveSolver("tableaus/rk45_table_v2.txt", epsilon)
    y0 = np.array([1., 0])

    xvals, tvals = rk_solver.ode_approx(circle_func, 0, 160, y0, 0.00125)
    points = xvals.T
    fig = plt.figure()
    plt.plot(points[0], points[1], marker='.')
    plt.show()


def test_flock():
    alpha = 1.6
    beta = 0.5
    C_a = 0.5
    ell_a = 2
    C_r = 1
    ell_r = 0.5
    N = 5
    tolerances = [10e-2, 10e-3, 10e-5, 10e-8]

    flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, N)
    epsilon = 10e-2
    rk_solver = adps.AdaptiveSolver("tables/rk45_table.txt", epsilon)
    y0 = init_flock(N, 0)
    xval_list = []
    tval_list = []
    times_list = []
    for epsilon in tolerances:
        solver = adps.AdaptiveSolver("tables/rk45_table.txt", epsilon)
        starttime = timeit.default_timer()
        xvals, tvals = solver.ode_approx(flock.f, 0, 30, y0, 0.00125)
        endtime = timeit.default_timer()
        times_list.append(endtime - starttime)
        xval_list.append(xvals)
        tval_list.append(tvals)
    # xvals, tvals = rk_solver.ode_approx(flock.f, 0, 30, y0, 0.00125)

    fig = plt.subplots(2, 2)
    plt.suptitle(f"D'Orsogna et al. model: 5 agents, t = 0 to 30\n", fontsize=14)
    plt.tight_layout()
    plt.subplot(2, 2, 1)
    points = xval_list[0].T
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.title(r"$\epsilon = 10^{-2}$:" + "{:.02f} sec".format(times_list[0]))
    plt.subplot(2, 2, 2)
    points = xval_list[1].T
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.title(r"$\epsilon = 10^{-3}$:" + "{:.02f} sec".format(times_list[1]))
    plt.subplot(2, 2, 3)
    points = xval_list[2].T
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.title(r"$\epsilon = 10^{-5}$:" + "{:.02f} sec".format(times_list[2]))
    plt.subplot(2, 2, 4)
    points = xval_list[3].T
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.title(r"$\epsilon = 10^{-8}$:" + "{:.02f} sec".format(times_list[3]))

    plt.show()

    # points = xvals.T
    # fig = plt.figure()
    # for i in range(N):
    #     plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    # plt.show()


if __name__ == '__main__':
    test_flock()

import adaptive_solver as adps
import numpy as np
import matplotlib.pyplot as plt
import models.dorsogna_flat as model
import models.cuckersmale_flat as cs_model
import timeit
import matplotlib.animation as ani


def circle_func(t, y):
    """
    :param t: independent (time) variable
    :param y: 2d numpy array
    :return: tangential velocity for circular motion
    """
    A = np.array([[0, -1], [1, 0]])
    return A @ y


def init_flock(N, seed, spread, max_vel):
    generator = np.random.RandomState(seed)
    positions = np.zeros([N, 4])
    for i in range(N):
        for j in range(2):
            positions[i][j] = generator.uniform(-spread, spread)
        for j in range(2, 4):
            positions[i][j] = generator.uniform(-max_vel, max_vel)
    return positions.reshape(N * 4)


def test_circle():
    epsilon = 10e-12
    rk_solver = adps.AdaptiveSolver("tableaus/rk45_table_v2.txt", epsilon)
    y0 = np.array([1., 0])

    xvals, tvals = rk_solver.ode_approx(circle_func, 0, 160, y0, 0.00125)
    points = xvals.T
    fig = plt.figure()
    plt.plot(points[0], points[1], marker='.')
    plt.show()


def test_flock(alpha, beta, C_r, C_a, ell_r, ell_a, N, b, y0):
    tolerances = [10e-2, 10e-3, 10e-5, 10e-8]

    do_flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, N)
    cs_flock = cs_model.CuckerSmaleModel(10, 2, 0.25, 5)
    epsilon = 10e-2
    rk_solver = adps.AdaptiveSolver("tables/rk45_table.txt", epsilon)
    xval_list = []
    tval_list = []
    times_list = []
    for epsilon in tolerances:
        solver = adps.AdaptiveSolver("tables/rk45_table.txt", epsilon)
        starttime = timeit.default_timer()
        xvals, tvals = solver.ode_approx(do_flock.f, 0, b, y0, 0.00125)
        endtime = timeit.default_timer()
        times_list.append(endtime - starttime)
        xval_list.append(xvals)
        tval_list.append(tvals)
    # xvals, tvals = rk_solver.ode_approx(flock.f, 0, 30, y0, 0.00125)

    # fig = plt.subplots(2, 2)
    # plt.suptitle("D'Orsogna et al. model: {0} agents, t = 0 to {1}\n".format(N, b), fontsize=14)
    # plt.tight_layout()
    # plt.subplot(2, 2, 1)
    # points = xval_list[0].T
    # for i in range(N):
    #     plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    # plt.title(r"$\epsilon = 10^{-2}$: " + "{:.02f} sec".format(times_list[0]))
    # plt.subplot(2, 2, 2)
    # points = xval_list[1].T
    # for i in range(N):
    #     plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    # plt.title(r"$\epsilon = 10^{-3}$: " + "{:.02f} sec".format(times_list[1]))
    # plt.subplot(2, 2, 3)
    # points = xval_list[2].T
    # for i in range(N):
    #     plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    # plt.title(r"$\epsilon = 10^{-5}$: " + "{:.02f} sec".format(times_list[2]))
    # plt.subplot(2, 2, 4)
    # points = xval_list[3].T
    # for i in range(N):
    #     plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    # plt.title(r"$\epsilon = 10^{-8}$: " + "{:.02f} sec".format(times_list[3]))

    # plt.show()

    points = xval_list[3].T
    radii = points[0] - points[4]
    fig = plt.figure()
    a = min(radii)
    for i in range(len(radii)):
        if radii[i] >= 2.133:
            print(i)
            break
    # plt.plot(tval_list[3], radii, marker='.')
    # plt.plot([0, 15], [2.133, 2.133])
    for i in range(N):
        plt.plot(points[4 * i], points[(4 * i) + 1], marker='.')
    plt.show()


def draw_frame(frame, N, points):
    for i in range(N):
        plt.plot(points[4 * i][frame - 5:frame], points[(4 * i) + 1][frame - 5:frame], marker='.')


if __name__ == '__main__':
    alpha = 1.6
    beta = 0.5

    C_r = 7.4 * 20
    C_a = 3.8 * 20
    ell_r = 0.7
    ell_a = 4.9

    N = 4
    b = 100
    initial_distance = 100

    # y0 = init_flock(N, 0, initial_distance, 0)
    # y0_list = [1.189927, 0, 0, 0,
    #            0, 0, 0, 0,
    #            -1.189927, 0, 0, 0]
    y0_list = [-1., 0, 0, -0.5,
               0, 1., -0.5, 0,
               1., 0, 0, 0.5,
               0, -1., 0.5, 0]
    y0 = np.array(y0_list)
    test_flock(alpha, beta, C_r, C_a, ell_r, ell_a, N, b, y0)

import timeit

import tableau_solver as btab
import numpy as np
import matplotlib.pyplot as plt
import models.dorsogna_flat as model
import matplotlib.gridspec as gridspec


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


def init_mill_flock(N, distance, speed):
    theta_step = 2*np.pi / N
    positions = np.empty([N, 4])
    for i in range(N):
        theta = i * theta_step
        vel_theta = theta + (np.pi / 2)
        positions[i] = [distance * np.cos(theta), distance * np.sin(theta), speed * np.cos(vel_theta), speed * np.sin(vel_theta)]
    return positions.reshape(N*4)


def plot_paths(points_list, N):
    fig = plt.figure(1)
    plt.title("RK4 Approximation")
    for i in range(N):
        plt.plot(points_list[0][4 * i], points_list[0][(4 * i) + 1], marker='.')
    plt.figure(2)
    plt.title("Euler Approximation")
    for i in range(N):
        plt.plot(points_list[1][4 * i], points_list[1][(4 * i) + 1])
    plt.show()


def plot_endpts(points_list, N):
    arrow_length = 0.40
    fig = plt.figure()

    colors = ['r', 'b']
    labels = ['RK4 Approx.', 'RK5 Approx.']
    for j in range(len(points_list)):
        points = points_list[j]
        color = colors[j % len(colors)]
        plt.title("D'Orsogna et al. model: t = 0 to 30")
        label = labels[j % len(labels)]
        for i in range(N):
            path_length = len(points[4*i])
            final_pos = np.array([points[4*i][path_length-1], points[(4*i)+1][path_length-1]])
            final_vel = np.array([points[(4*i)+2][path_length-1], points[(4*i)+3][path_length-1]])
            final_vel = arrow_length * (final_vel / np.linalg.norm(final_vel))
            if i == 0:
                plt.plot(points[4 * i][path_length - 1:], points[(4 * i) + 1][path_length - 1:], color=color,
                         marker='o', label=label)
            else:
                plt.plot(points[4 * i][path_length - 1:], points[(4 * i) + 1][path_length - 1:], color=color,
                         marker='o')
            plt.arrow(final_pos[0], final_pos[1], final_vel[0], final_vel[1], length_includes_head=True,
                      head_width=arrow_length*0.20)
            plt.legend()
    plt.show()


def plot_endpts_grid(points_list, N):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    main_plot = fig.add_subplot(gs[0, :])
    arrow_length = 0.40

    colors = ['r', 'b']
    labels = ['RK4 Approx.', 'RK5 Approx.']
    plt.suptitle("D'Orsogna et al. model: t = 0 to 30, N = 7")
    for j in range(len(points_list)):
        ax = fig.add_subplot(gs[1, j])
        points = points_list[j]
        color = colors[j % len(colors)]
        label = labels[j % len(labels)]
        for i in range(N):
            path_length = len(points[4*i])
            final_pos = np.array([points[4*i][path_length-1], points[(4*i)+1][path_length-1]])
            final_vel = np.array([points[(4*i)+2][path_length-1], points[(4*i)+3][path_length-1]])
            final_vel = arrow_length * (final_vel / np.linalg.norm(final_vel))
            if i == 0:
                main_plot.plot(points[4 * i][path_length - 1:], points[(4 * i) + 1][path_length - 1:], color=color,
                         marker='o', label=label)
            else:
                main_plot.plot(points[4 * i][path_length - 1:], points[(4 * i) + 1][path_length - 1:], color=color,
                         marker='o')
            ax.plot(points[4 * i][path_length - 1:], points[(4 * i) + 1][path_length - 1:], color=color,
                           marker='o')
            plt.arrow(final_pos[0], final_pos[1], final_vel[0], final_vel[1], length_includes_head=True,
                      head_width=arrow_length*0.20)
            main_plot.legend()
    plt.show()


def graph_euler_instability():
    alpha = 1.6
    beta = 0.5

    C_r = 7.4 * 20
    C_a = 3.8 * 20
    ell_r = 0.7
    ell_a = 4.9

    N = 2
    b = 6
    h_large = 1.0
    h_small = 0.001
    r_max = 2.133
    euler_solver = btab.TableauSolver("tables/euler_tableau.txt")

    flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, N)
    y0 = np.array([r_max / 2, 0, -r_max, 0,
                    -r_max / 2, 0, r_max, 0])
    bad_yvals, bad_tvals = euler_solver.ode_approx(flock.f, 0, b, y0, h_large)
    good_yvals, good_tvals = euler_solver.ode_approx(flock.f, 0, b, y0, h_small)

    fig = plt.figure()
    plt.title("Euler Approximation, t = 0 to 6")
    radii_bad = bad_yvals.T[0] - bad_yvals.T[4]
    radii_good = good_yvals.T[0] - good_yvals.T[4]
    plt.ylabel(r"$x_1 - x_2$")
    plt.xlabel("t")
    plt.ylim(-7, 5.25)
    plt.plot(bad_tvals, radii_bad, label="h = 1.0")
    plt.plot(good_tvals, radii_good, label="h = 0.001")
    plt.legend()
    plt.show()


def graph_convergence():
    alpha = 1.6
    beta = 0.5

    C_r = 7.4 * 20
    C_a = 3.8 * 20
    ell_r = 0.7
    ell_a = 4.9

    N = 8
    b = 120
    rk4_solver = btab.TableauSolver("tables/rk4_tableau.txt")
    rk5_solver = btab.TableauSolver("tables/rk5_tableau.txt")
    euler_solver = btab.TableauSolver("tables/euler_tableau.txt")

    # step_sizes = np.arange(1.0, 0, -0.1)
    step_sizes = np.array([0.003])
    # N_vals = np.arange(11, 16, 1)
    N_vals = np.array([20])

    # This is the code used to produce the convergence data.
    # errors = []
    data_file = open('data/euler_output_runtime_N_v2.txt', 'w')
    for N_val in N_vals:
        flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, N_val)
        y0 = init_mill_flock(N_val, 1.0, 0.5)
        rk4_start_time = timeit.default_timer()
        xvals, tvals = rk4_solver.ode_approx(flock.f, 0, b, y0, 0.05)
        rk4_total_time = timeit.default_timer() - rk4_start_time
        rk5_start_time = timeit.default_timer()
        # ref_vals, ref_tvals = rk5_solver.ode_approx(flock.f, 0, b, y0, 0.05)
        rk5_total_time = timeit.default_timer() - rk5_start_time
        #     # print("Runtime for RK4 approximation: {0} seconds\n".format(endtime - starttime))
        end_approx = xvals[len(xvals) - 1]
        # end_expected = ref_vals[len(ref_vals)-1]
        # error = np.linalg.norm(end_approx - end_expected)
        print("N = {0}, Rk4 runtime: {1}, RK5 runtime: {2}".format(N_val, rk4_total_time, rk5_total_time))
        #     # print("Step size = {0}, t = {1}, err = {6}:\n\trk4: computation time: {2} "
        #     #       "\n\trk4: end state: {3}\n\trk5: computation time: {4}\n\trk5: end state: {5}\n\n".format(step_size, b,
        #     #       0, 0, 0, [0], 0))
        data_file.write("{0}\n".format(N_val))
        data_file.write("{0}\n".format(rk4_total_time))
        data_file.write("{0}\n".format(0))
    #     # for number in end_expected:
    #     #     data_file.write("{0} ".format(number))
    #     data_file.write("\n")
    data_file.close()
    # print("Magnitude of difference for stepsize {0}: {1}".format(step_size, ))
    # plt.plot(n_vals, errors)
    # points = xvals.T
    # euler_x, euler_t = euler_solver.ode_approx(flock.f, 0, 40, y0, 0.0125)
    # euler_points = euler_x.T

    # last_N = 7
    # flock = model.DorsognaModel(alpha, beta, C_a, ell_a, C_r, ell_r, last_N)
    # y0 = init_mill_flock(last_N, 1.0, 0.5)
    # xvals, tvals = rk4_solver.ode_approx(flock.f, 0, b, y0, 0.05)
    # ref_vals, ref_tvals = rk5_solver.ode_approx(flock.f, 0, b, y0, 0.05)
    # plot_endpts_grid([xvals.T, ref_vals.T], last_N)

    # plot_paths([points, euler_points])


if __name__ == '__main__':
    graph_euler_instability()



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

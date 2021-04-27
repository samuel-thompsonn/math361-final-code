import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath):
    data = open(filepath)
    stepsizes = []
    datavalues = []
    runtimes = []
    while True:
        stepsize = data.readline()
        if stepsize == '':
            break
        stepsizes.append(float(stepsize))

        runtime = data.readline()
        runtimes.append(float(runtime))
        data_vals = data.readline()
        data_arr = np.empty(32)
        values = data_vals.split()
        for i in range(32):
            data_arr[i] = (float(values[i]))
        datavalues.append(data_arr)
    data.close()
    return stepsizes, datavalues, runtimes


def graph_convergence():
    rk4_stepsizes, rk4_datavalues, runtimes = read_data('data/output_rk4.txt')
    exp_stepsizes, exp_datavalues, runtimes = read_data('data/output_rk5.txt')
    expected_endpt = exp_datavalues[0]
    errors = []

    for val_vector in rk4_datavalues:
        errors.append(np.linalg.norm(val_vector - expected_endpt))

    plt.figure()
    plt.plot(np.log(rk4_stepsizes), errors)
    plt.show()


def graph_timestep_runtime():
    rk4_stepsizes, rk4_datavalues, runtimes = read_data('data/output_rk4.txt')
    plt.figure()
    plt.title("RK4 Runtime")
    plt.xlabel(r"$\log(1/h)$")
    plt.ylabel("seconds")
    stepsizes = np.array(rk4_stepsizes)
    plt.plot(np.log(stepsizes**(-1)), runtimes)
    plt.plot(np.log(stepsizes**(-1)), 0.25*stepsizes**(-1))
    # It is extremely extremely obvious that the runtime would be proportional to 1 / stepsize,
    # since that's just linear with the number of time steps that  we calculate, since each timestep
    # has the same runtime!! you fool


if __name__ == '__main__':
    data = open('data/rk4_output_runtime_N.txt').readlines()
    N_vals = []
    runtime_vals = []
    for i in range(int(len(data) / 4)):
        N_vals.append(int(data[4*i]))
        runtime_vals.append(float(data[(4*i)+1]))
    fig = plt.figure()
    plt.title("RK4 Runtime: t = 0 to 120, h = 0.05")
    fig_N_vals = np.array(N_vals)
    plt.xlabel(r"number of agents $N$")
    plt.ylabel("seconds")
    plt.plot(fig_N_vals, runtime_vals, label="RK4 Runtime")
    plt.plot(fig_N_vals, (1/4)*(fig_N_vals**2), label=r"$\frac{N^2}{4}$")
    plt.xticks(np.arange(0, 21, 4))
    plt.legend()

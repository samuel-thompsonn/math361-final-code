import numpy as np
import matplotlib.pyplot as plt


def morse_potential(r, C_r, C_a, ell_r, ell_a):
    return -C_a*np.exp(-r/ell_a) + C_r*np.exp(-r/ell_r)


def morse_gradient(r, C_r, C_a, ell_r, ell_a):
    return (1/ell_a)*C_a*np.exp(-r/ell_a) - (1/ell_r)*C_r*np.exp(-r/ell_r)


if __name__ == '__main__':
    r = np.arange(-2, 10, 0.05)
    C_r = 7.4*20
    C_a = 3.8*20
    ell_r = 0.7
    ell_a = 4.9
    morse_p = morse_potential(r, C_r, C_a, ell_r, ell_a)
    morse_d = morse_gradient(r, C_r, C_a, ell_r, ell_a)

    y_min = -50
    y_max = 60
    fig = plt.figure()
    plt.title(r"Morse potential: C = 1.947, $\ell = 0.143$")
    plt.ylim(y_min, y_max)
    plt.xlim(-1, 8)
    plt.plot([0, 0], [y_min, y_max], 'k')
    plt.plot([-1, 8], [0, 0], 'k')
    plt.plot(r, morse_p, label="U(r)")
    plt.plot(r, morse_d, '--', label=r"$\frac{dU}{dr}(r)$")
    plt.xlabel(r"r = $|x_i - x_j|$")
    plt.legend()
    plt.show()

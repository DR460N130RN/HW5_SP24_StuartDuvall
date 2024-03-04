import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def ode_system(t, X, *params):
    """
    Defines the ODE system in terms of state variables
    I have as unknowns:
    x: position of the piston (This is not strictly needed unless I want to know x(t))
    xdot: velocity of the piston
    p1: pressure on right of piston
    p2: pressure on left of the piston
    For initial conditions, we see: x=x0=0, xdot=0, p1=p1_0=p_a, p2=p2_0=p_a
    :param t:
    :param X:
    :param params:
    :return:
    """
    A, Cd, Ps, Pa, V, beta, rho, Kvalve, m, y = params
    x, xdot, p1, p2 = X

    xddot = (1 / m) * (A * p2 - Cd * rho * A * abs(xdot) * xdot - Ps * A * np.exp(-beta * x) - Kvalve * (p1 - p2))
    p1dot = (y / V) * (p2 - p1)
    p2dot = (1 / V) * (Kvalve * (p1 - p2) - Cd * rho * A * abs(xdot) * xdot)

    return [xdot, xddot, p1dot, p2dot]


def main():
    t = np.linspace(0, 0.02, 200)
    params = (4.909E-4, 0.6, 1.4E7, 1.0E5, 1.473E-4, 2.0E9, 850.0, 2.0E-5, 30, 0.002)
    pa = params[3]
    ic = [0, 0, pa, pa]

    sln = solve_ivp(ode_system, (0, 0.02), ic, args=params, t_eval=t)

    xvals = sln.y[0]
    xdot = sln.y[1]
    p1 = sln.y[2]
    p2 = sln.y[3]

    plt.subplot(2, 1, 1)
    plt.plot(t, xvals, 'r-', label='$x$')
    plt.ylabel('$x$')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    plt.plot(t, xdot, 'b-', label='$\dot{x}$')
    plt.ylabel('$\dot{x}$')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(t, p1, 'b-', label='$P_1$')
    plt.plot(t, p2, 'r-', label='$P_2$')
    plt.legend(loc='lower right')
    plt.xlabel('Time, s')
    plt.ylabel('$P_1, P_2 (Pa)$')

    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def ff(Re, rr, CBEQN=False):
    """
    Calculates the friction factor of a pipe
    based on the type of flow
    :param Re:
    :param rr:
    :param CBEQN:
    :return:
    """
    if CBEQN:
        cb = lambda f: 1 / np.sqrt(f) + 2 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))
        result = fsolve(cb, 0.01)
        return result[0]
    else:
        return 64 / Re

def plotMoody(plotPoint=False, pt=(0, 0)):
    """
    Produces the Moody diagram for a Reynolds number ranging
    from 1 to 10^8 and for a relative roughness ranging
    from 0 to 0.05 with 20 steps.
    :param plotPoint:
    :param pt:
    :return:
    """
    ReValsCB = np.logspace(3.6, 8, 20)
    ReValsL = np.logspace(np.log10(600.0), np.log10(2000.0), 20)
    ReValsTrans = np.logspace(3.3, 3.6, 20)
    rrVals = np.array([0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4, 1E-3, 2E-3, 4E-3, 6E-3, 8E-8, 1.5E-2, 2E-2, 3E-2, 4E-2, 5E-2])

    ffLam = np.array([ff(Re, 0) for Re in ReValsL])
    ffTrans = np.array([ff(Re, 0) for Re in ReValsTrans])
    ffCB = np.array([[ff(Re, rr) for Re in ReValsCB] for rr in rrVals])

    plt.loglog(ReValsL, ffLam, 'b-', label="Laminar")
    plt.loglog(ReValsTrans, ffTrans, 'r--', label="Transition")
    for i in range(len(rrVals)):
        plt.loglog(ReValsCB, ffCB[i], label=str(rrVals[i]))

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re$", fontsize=16)
    plt.ylabel(r"Friction factor $f$", fontsize=16)
    plt.text(2.5E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.tick_params(axis='both', grid_linewidth=1, grid_linestyle='solid', grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')
    if plotPoint:
        plt.plot(pt[0], pt[1], 'ro', markersize=12, markeredgecolor='red', markerfacecolor='none')

    plt.legend(title="Relative Roughness", loc="best")
    plt.show()

def main():
    plotMoody()

if __name__ == "__main__":
    main()

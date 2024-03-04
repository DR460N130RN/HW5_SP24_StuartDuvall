import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from math import floor, ceil

def RSquared(x, y, coeff):
    """
    Calculates the R^2 value
    :param x:
    :param y:
    :param coeff:
    :return:
    """
    AvgY = np.mean(y)
    SSTot = np.sum((y - AvgY) ** 2)
    SSRes = np.sum((y - Poly(x, *coeff)) ** 2)
    RSq = 1 - (SSRes / SSTot)
    return RSq

def Poly(xdata, *a):
    """
    Calculates the value for a polynomial
    :param xdata:
    :param a:
    :return:
    """
    y = np.zeros_like(xdata)
    power = len(a) - 1
    for i in range(power + 1):
        y += a[i] * (xdata ** i)
    return y

def PlotLeastSquares(x, y, coeff, showpoints=True, npoints=500):
    """
    Makes a plot for a polynomial
    :param x:
    :param y:
    :param coeff:
    :param showpoints:
    :param npoints:
    :return:
    """
    Xmin, Xmax = min(x), max(x)
    Ymin, Ymax = min(y), max(y)

    if len(coeff) == 0:
        coeff = LeastSquaresFit(x, y, 1)

    xvals = np.linspace(Xmin, Xmax, npoints)
    yvals = Poly(xvals, *coeff)

    RSq = RSquared(x, y, coeff)

    plt.plot(xvals, yvals, linestyle='dashed', color='black', linewidth=2)
    plt.title(r'$R^2={:0.3f}$'.format(RSq))
    plt.xlim(floor(Xmin * 10) / 10, ceil(Xmax * 10) / 10)
    plt.ylim(floor(Ymin), ceil(Ymax * 10) / 10)
    if showpoints:
        plt.plot(x, y, linestyle='none', marker='o', markerfacecolor='white', markeredgecolor='black', markersize=10)
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.gca().tick_params(axis='both', top=True, right=True, direction='in', grid_linewidth=1, grid_linestyle='dashed', grid_alpha=0.5)
    plt.show()
    return xvals, yvals

def LeastSquaresFit(x, y, power=1):
    """
    fits some x, y data with a polynomial
    :param x:
    :param y:
    :param power:
    :return:
    """
    coeff, cov = curve_fit(Poly, x, y, p0=[1] * (power + 1))
    return coeff

def main():
    x = np.array([0.05, 0.11, 0.15, 0.31, 0.46, 0.52, 0.70, 0.74, 0.82, 0.98, 1.17])
    y = np.array([0.956, 1.09, 1.332, 0.717, 0.771, 0.539, 0.378, 0.370, 0.306, 0.242, 0.104])

    coeff1 = LeastSquaresFit(x, y, 1)
    linx, liny = PlotLeastSquares(x, y, coeff1, showpoints=True, npoints=500)
    RSqLin = RSquared(x, y, coeff1)

    coeff2 = LeastSquaresFit(x, y, 3)
    cubx, cuby = PlotLeastSquares(x, y, coeff2, showpoints=True, npoints=500)
    RSqCub = RSquared(x, y, coeff2)

    plt.plot(linx, liny, linewidth=2, linestyle='dashed', color='black', label=r'Linear fit ($R^2={:0.3f}$)'.format(RSqLin))
    plt.plot(cubx, cuby, linewidth=2, linestyle='dotted', color='black', label='Cubic fit ($R^2={:0.3f}$)'.format(RSqCub))
    plt.plot(x, y, linestyle='none', marker='o', markersize=10, markerfacecolor='white', markeredgecolor='black', label='Data')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()

    plt.tick_params(axis='both', top=True, right=True, direction='in', grid_linewidth=1, grid_linestyle='dashed', grid_alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()

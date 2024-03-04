import Prob_a as pta
import random as rnd
from matplotlib import pyplot as plt

def ffPoint(Re, rr):
    """
    Outputs a friction factor using Re and rr as parameters as well as the following;
    if Re>4000 use Colebrook Equation
    if Re<2000 use f=64/Re
    else calculate a probabilistic friction factor where the distribution has a mean midway between the prediction
    of the f=64/Re and Colebrook Equations and a standard deviation of 20% of this mean
    :param Re:
    :param rr:
    :return:
    """
    if Re >= 4000:
        return pta.ff(Re, rr, CBEQN=True)
    elif Re <= 2000:
        return pta.ff(Re, rr)
    else:
        CBff = pta.ff(Re, rr, CBEQN=True)
        Lamff = pta.ff(Re, rr)
        mean = (CBff + Lamff) / 2
        sig = 0.2 * mean
        return rnd.normalvariate(mean, sig)

def PlotPoint(Re, f):
    pta.plotMoody(plotPoint=True, pt=(Re, f))

def main():
    Re = float(input("Enter the Reynolds number: "))
    rr = float(input("Enter the relative roughness: "))
    f = ffPoint(Re, rr)
    PlotPoint(Re, f)

if __name__ == "__main__":
    main()

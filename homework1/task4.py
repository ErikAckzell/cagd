from scipy.special import comb
import numpy
import pylab

def bernsteinPol(n, t0):
    """
    Constructs and evaluates the Bernstein Polynomials of a given degree at point t0.
    :param n: degree
    :param t0: value of t
    :return: list of values of the polynomials at t=t0
    """
    return [comb(n,j)*(1-t0)**(n-j)*t0**j for j in range(n+1)]

def plotBernstein(n,t0=0,t1=1,steps=30):
    """
    Plots the Bernstein Polynomials of any given degree.
    :param n: degree
    :param t0: start point of interval
    :param t1: end point of interval
    :param steps: number of grid points to plot
    """
    # Construct the grid of t values
    tgrid = numpy.linspace(t0,t1,steps)

    # Evaluate the Bernstein polynomials at each grid point
    ugrid = zip(*[bernsteinPol(n,t) for t in tgrid])

    # Plot each polynomial
    for ind, i in enumerate(ugrid):
        pylab.plot(tgrid,i,label='$j={}$'.format(ind))
    pylab.title('Bernstein Polynomials of Degree {}'.format(n))
    pylab.xlabel('$t$')
    pylab.ylabel('$B^{}_j(t)$'.format(n))
    pylab.legend(loc = 'upper right')
    pylab.show()

if __name__ == '__main__':
    plotBernstein(4)
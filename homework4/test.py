import scipy
from matplotlib import pyplot as plt
import numpy as np


class Bspline(object):
    def __init__(self, grid, controlpoints, degree, dim):
        """
        grid (iterable): grid points should have multiplicity 3 in order to
        have the spline starting and ending in the first and last control
        point, respectively.
        controlpoints (array): should be on the form
                controlpoints = array([
                                      [d_00, d01],
                                      [d_10, d11],
                                      ...
                                      [d_L0, d_L1],
                                      ]),
        i.e. an (L+1)x2 array.
        """
        try:
            grid = scipy.array(grid)
            grid = grid.reshape((len(grid), 1))
        except ValueError:
            raise ValueError('Grid should be a one-dimensional list or array')
        if controlpoints.shape[1] != 2:
            raise ValueError('Controlpoints should be an (L+1)x2 array.')
        self.grid = grid.reshape((len(grid), 1))
        self.controlpoints = controlpoints
        self.degree = degree
        self.dim = dim

    def __call__(self, u):
        index = self.get_index(u)
        r = self.get_mult(index, u)
        current_controlpoints = self.get_controlpoints(index, r)
        print('index:', index,'r:', r)
        d = scipy.zeros((len(current_controlpoints), len(current_controlpoints), self.dim))
        d[0] = current_controlpoints
        print(scipy.shape(d), d)
        for s in range(1, self.degree - r): # columns SHOULD IT BE +1 AT THE END?
            for j in range(index - self.degree + s, index - r): # rows SHOULD IT BE +1 AT THE END?
                a = (u - self.grid[j])/(self.grid[j + self.degree - s + 1] - self.grid[j])
                print('j:', j, 's:', s)
                d[j,s] = (1-a)*d[j-1,s-1] + a*d[j,s-1]
        return d[index - r - 1, self.degree - r ]

    def get_mult(self, index, u):
        if u == self.grid[index]:
            return len([i for i in self.grid if i == self.grid[index]])
        else:
            return 0

    def get_controlpoints(self, index, r):
        """
        Method to obtain the current control points, d_{i-n}, ..., d_i, for de Boor's algorithm, where n is the degree
        and i is the index of the interval for which u lies in: [u_i,u_{i+1}). If u=u_i and u_i has multiplicity r the
        current control points changes to be d_{i-n},...,d_{i-r-1},d_{i-r}.
        index (int): the index depending on the point u at which to evaluate
        the spline (see get_index method).
        """
        # Assert an error if the degree is bigger than the index, which only happens if the curve is not clamped
        assert (index - self.degree) >= 0, 'The curve is not clamped.'
        if r > self.degree:
            # We are only working with clamped curves, therefore there are always have n+1 knots at the endpoints
            # A knot can not have multiciply higher than the degree unless it is at the end points
            current_controlpoints = self.controlpoints[:self.degree + 1]
        else: # r == 0:
            current_controlpoints = self.controlpoints[index - self.degree:index + 1]

        #else:
         #   current_controlpoints = self.controlpoints[index - self.degree:index - r + 1]
        return current_controlpoints


    def get_index(self, u):
        """
        Method to get the index of the grid point at the left endpoint of the
        gridpoint interval at which the current value u is. If u belongs to
            [u_I, u_{I+1}]
        it returns the index I.
        u (float): value at which to evaluate the spline
        """
        if u == self.grid[-1]:  # check if u equals last knot
            index = (self.grid < u).argmin() - 1
        else:
            index = (self.grid > u).argmax() - 1
        return index

    def plot(self, title, filename=None, points=300, controlpoints=True, markSeq=False, clamped=False):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the controlpoints as well
        markSeq (bool): if true, marks each spline sequence in the plot
        clamped (bool): if true, skips the multiples in the endpoints when each sequence is marked
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if markSeq:
            if clamped:
                start, end = self.degree, len(self.grid) - self.degree - 1
            else:
                start, end = 0, len(self.grid) - 1
            for i in range(start, end):
                ulist = scipy.linspace(self.grid[i], self.grid[i+1], points)
                ax.plot(*zip(*[self(u) for u in ulist]), label='B-Spline between knots {} and {}'.format(self.grid[i],
                                                                                                          self.grid[i+1]
                                                                                                          ))
        else:
            # list of u values for which to plot
            ulist = scipy.linspace(self.grid[0], self.grid[-1], points)
            ax.plot(*zip(*[self(u) for u in ulist]), label='B-Spline Curve')
        if controlpoints:  # checking whether to plot control points
            ax.plot(*zip(*self.controlpoints), 'o--', label='Control Points')

        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.title(title)
        plt.show()
        if filename:
            fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == '__main__':

    ### Task 2 ###
    numPoints = 200
    grid = scipy.array([1,1,1,1,6/5,7/5,8/5,9/5,2,2,2,2])
    controlpoints = scipy.array([[0.7,-0.4],
                                [1.0,-0.4],
                                [2.5,-1.2],
                                [3.2,-0.5],
                                [-0.2,-0.5],
                                [0.5,-1.2],
                                [2.0,-0.4],
                                [2.3,-0.4]])
    bspline = Bspline(grid, controlpoints, 3, 2)
    bspline.plot('test', markSeq=True, clamped=True)
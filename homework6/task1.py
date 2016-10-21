import scipy
from matplotlib import pyplot as plt
import numpy as np
import math


class Bspline(object):
    def __init__(self, grid, controlpoints, degree):
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
        self.grid = grid.reshape((len(grid), 1))
        self.controlpoints = controlpoints
        self.degree = degree
        self.dim = scipy.shape(controlpoints)[1]

    def __call__(self, u):
        d = self.get_deBoor_array(u)
        return d[-1, -1]

    def get_deBoor_array(self, u):
        index = self.get_index(u)
        r = self.get_mult(index, u)
        current_controlpoints = self.get_controlpoints(index, r, u)
        num_controlpoints = len(current_controlpoints)
        d = scipy.zeros((num_controlpoints, num_controlpoints, self.dim))
        d[0] = current_controlpoints
        for s in range(1, num_controlpoints): # column
            for j in range(s, num_controlpoints): # rows
                left = index - self.degree + j
                right = index + j - s + 1
                a = (u - self.grid[left])\
                    / (self.grid[right] - self.grid[left])
                d[s,j] = (1-a)*d[s-1,j-1] + a*d[s-1,j]
        return d

    def get_mult(self, index, u):
        if u == self.grid[index]:
            return len([i for i in self.grid if i == self.grid[index]])
        elif u == self.grid[-1]:
            return len([i for i in self.grid if i == self.grid[-1]])
        else:
            return 0

    def get_controlpoints(self, index, r, u):
        """
        Method to obtain the current control points, d_{i-n}, ..., d_i, for de Boor's algorithm, where n is the degree
        and i is the index of the interval for which u lies in: [u_i,u_{i+1}). If u=u_i and u_i has multiplicity r the
        current control points changes to be d_{i-n},...,d_{i-r-1},d_{i-r}.
        index (int): the index depending on the point u at which to evaluate
        the spline (see get_index method).
        """
        if r > self.degree:
            # The curve is clamped if we have n+1 knots at the endpoints
            # A knot can not have multiciply higher than the degree unless it is at the end points
            if u == self.grid[0]:
                # startpoint
                current_controlpoints = self.controlpoints[:self.degree + 1]
            else:
                # endpoint
                current_controlpoints = self.controlpoints[-(self.degree + 1):]
        else:
            current_controlpoints = self.controlpoints[index - self.degree:index - r + 1]
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

    def plot(self, title, filename=None, points=300, controlpoints=True,
             markSeq=False, clamped=False):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the control points as well
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
            if clamped:
                ulist = scipy.linspace(self.grid[0], self.grid[-1], points)
            else:
                ulist = scipy.linspace(self.grid[self.degree], self.grid[-self.degree], points)
            ax.plot(*zip(*[self(u) for u in ulist]), label='B-Spline Curve')
        if controlpoints:  # checking whether to plot control points
            ax.plot(*zip(*self.controlpoints), 'o--', label='Control Points')

        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        ax.set_title(title)
        if filename:
            fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()


class beziercurve(object):
    """
    This is a class for Bézier curves.
    """

    def __init__(self, controlpoints):
        """
        An object of the class is initialized with a set of control points in
        the plane.
        """
        self.controlpoints = controlpoints
        self.xlow = min(self.controlpoints[:, 0])
        self.xhigh = max(self.controlpoints[:, 0])
        self.ylow = min(self.controlpoints[:, 1])
        self.yhigh = max(self.controlpoints[:, 1])

    def __call__(self, t):
        """
        This method returns the point on the line for some t.
        """
        deCasteljauArray = self.get_deCasteljauArray(t)
        return deCasteljauArray[-1, -2:]

    def subdivision(self, t):
        """
        This method implements subdivision at t.
        """
        # getting the de Casteljau array using t
        deCasteljauArray = self.get_deCasteljauArray(t)
        # extracting the new controlpoints from the array
        controlpoints1 = scipy.array([deCasteljauArray[i, 2 * i:2 * i + 2]
                                      for i in range(len(self.controlpoints))])
        controlpoints2 = scipy.array([deCasteljauArray[-1, 2 * i:2 * i + 2]
                                      for i in range(len(self.controlpoints))])
        controlpoints2 = controlpoints2[::-1]
        curve1 = beziercurve(controlpoints1)
        curve2 = beziercurve(controlpoints2)

        return (curve1, curve2)

    def get_deCasteljauArray(self, t):
        """
        This method calculates and returns a matrix with the lower left corner
        containing the de Casteljau array, calculated for the specified t.
        """
        # initializing the array
        deCasteljauArray = scipy.column_stack((
            np.copy(self.controlpoints),
            scipy.zeros((len(self.controlpoints),
                         2 * len(self.controlpoints) - 2))
        ))
        # filling the array
        for i in range(1, len(deCasteljauArray)):
            for j in range(1, i + 1):
                deCasteljauArray[i, j * 2:j * 2 + 2] = (
                    (1 - t) * deCasteljauArray[i - 1, (j - 1) * 2:(j - 1) * 2 + 2] +
                    t * deCasteljauArray[i, (j - 1) * 2:(j - 1) * 2 + 2])
        return deCasteljauArray


if __name__ == '__main__':
#    """
    ### Task 1 ###
    D_initial = scipy.array([[0.7, -0.4],
                             [1.0, -0.4],
                             [2.5, -1.2],
                             [3.2, -0.5],
                             [-0.2, -0.5],
                             [0.5, -1.2],
                             [2.0, -0.4],
                             [2.3, -0.4]])
    r = 0.1
    D = scipy.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            alpha = 2 * scipy.pi * j / 7
            D[i, j] = scipy.array([D_initial[i, 0],
                                   D_initial[i, 1] + r * scipy.sin(alpha),
                                   r * scipy.cos(alpha)])
    uknots = scipy.array([1, 1, 1, 1, 6 / 5, 7 / 5, 8 / 5, 9 / 5, 2, 2, 2, 2])
    vknots = scipy.copy(uknots)
    degree = 3
    ugrid = scipy.linspace(uknots[0], uknots[-1], 200)
    vgrid = scipy.copy(ugrid)

#    """






















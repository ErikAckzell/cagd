import scipy
from matplotlib import pyplot as plt
import numpy as np


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
        if controlpoints.shape[1] != 2:
            raise ValueError('Controlpoints should be an (L+1)x2 array.')
        self.grid = grid.reshape((len(grid), 1))
        self.controlpoints = controlpoints
        self.degree = degree

    def __call__(self, u):
        """
        Method to evaluate the spline at point u, using de Boor's algorithm.
        """
        # get index of grid point left of u
        index = self.get_index(u)
        # get current controlpoints
        current_controlpoints = self.get_controlpoints(index)
        # setup matrix to store the values in the de Boor array:
        # deBoorvalues =
        #              d[I-2, I-1, I]
        #              d[I-1, I, I+1]   d[u, I-1, I]
        #              d[I, I+1, I+2]   d[u, I, I+1]   d[u, u, I]
        #              d[I+1, I+2, I+3] d[u, I+1, I+2] d[u, u, I+1] d[u, u, u]
        deBoorvalues = scipy.column_stack((current_controlpoints,
                                           scipy.zeros((4, 6))))

        # calculate values for de Boor array
        for i in range(1, 4): #rows
            for j in range(1, i + 1): #columns
                leftmostknot = index + i - 3  # current leftmost knot
                rightmostknot = leftmostknot + 4 - j  # current rightmost knot
                alpha = self.get_alpha(u, [leftmostknot, rightmostknot])
                deBoorvalues[i, j*2:j*2+2] = (
                            alpha * deBoorvalues[i-1, (j-1)*2:(j-1)*2+2] +
                            (1 - alpha) * deBoorvalues[i, (j-1)*2:(j-1)*2+2]
                                             )
        return deBoorvalues[3, -2:]

    def get_controlpoints(self, index):
        """
        Method to obtain the current control points for de Boor's algorithm.
        index (int): the index depending on the point u at which to evaluate
        the spline (see get_index method).
        """
        if index < 2:  # is index in very beginning
            current_controlpoints = self.controlpoints[0:4]  # use first points
        elif index > len(self.controlpoints) - 2:  # is index in very end
            current_controlpoints = self.controlpoints[-4:]  # use last points
        else:
            current_controlpoints = self.controlpoints[index - 2:index + 2]
        return current_controlpoints

    def get_alpha(self, u, indices):
        """
        Returns the alpha parameter used for linear interpolation of the
        values in the de Boor scheme.
        u (float): value at which to evaluate the spline
        indices (iterable): indices for the leftmost and rightmost knots
        corresponding to the current blossom pair
        """

        try:
            alpha = ((self.grid[indices[1]] - u) /
                     (self.grid[indices[1]] - self.grid[indices[0]]))
        except ZeroDivisionError:  # catch multiplicity of knots
            alpha = 0
        return alpha

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

    def plot(self, title, filename, points=300, controlpoints=True, markSeq=False, clamped=False):
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
        fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == '__main__':

    ### Task 2 ###
    grid = scipy.array([1,1,1,1,6/5,7/5,8/5,9/5,2,2,2,2])
    controlpoints = scipy.array([[0.7,-0.4],
                                [1.0,-0.4],
                                [2.5,-1.2],
                                [3.2,-0.5],
                                [-0.2,-0.5],
                                [0.5,-1.2],
                                [2.0,-0.4],
                                [2.3,-0.4]])
    bspline = Bspline(grid, controlpoints, 3)
    title = 'B-spline Curve Defined by the Grid $\{1,1,1,1,6/5,7/5,8/5,9/5,2,2,2,2\}$ \n and Control Points Marked on the Curve.'
    filename = 'task2'
    bspline.plot(title, filename, markSeq=True, clamped=True)

    """
    ### Task 4 ###
    grid = scipy.array([0,0,0,0,0,1/3,2/3,1,1,1,1,1])
    controlpoints = scipy.array([[0,0],
                                [-4,0],
                                [-5,2],
                                [-4,4.5],
                                [-2,5],
                                [1,5.5],
                                [1,0]])
    bspline = Bspline(grid, controlpoints, 4)
    title = 'B-spline Curve Defined by the Grid $\{0,0,0,0,0,1/3,2/3,1,1,1,1,1\}$ \n and Control Points Marked on the Curve.'
    bspline.plot(title)
    """
    """
    ### Task 5 ###
    grid = scipy.array([0,1/11,2/11,3/11,4/11,5/11,6/11,7/11,8/11,9/11,10/11,1])
    controlpoints = scipy.array([[0,0],
                                 [3,2],
                                 [9,-2],
                                 [7,-5],
                                 [1,-3],
                                 [1,-1],
                                 [3,1],
                                 [9,-1]])
    #for i in range(len(controlpoints) - 1):
     #   if i > 0:
      #      controlpoints[-i] = controlpoints[i-1]
       #     print('i=', i, controlpoints)
    bspline = Bspline(grid, controlpoints, 3)
    title = 'B-spline Curve Defined by the Grid $\{0,1/11,2/11,3/11,4/11,5/11,6/11,7/11,8/11,9/11,10/11,1\}$ \n and ' \
            'Control Points Marked on the Curve.'
    bspline.plot(title)
    """
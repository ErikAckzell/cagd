import scipy
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RationalBezierCurve(object):
    """
    This is a class for Bézier curves.
    """
    def __init__(self, controlpoints, weights):
        """
        An object of the class is initialized with a set of control points in
        the plane.
        """
        self.weights = weights
        self.original_controlpoints = controlpoints
        self.controlpoints = self.create_new_controlpoints(controlpoints)


    def __call__(self, t):
        """
        This method returns the point on the line for some t.
        """
        deCasteljauArray = self.get_deCasteljauArray(t)
        # turn into homogeneous coordinates
        point = deCasteljauArray[-1, -1]
        return 1/point[-1] * point

    def create_new_controlpoints(self, controlpoints):
        return scipy.array([numpy.append(controlpoints[i]*self.weights[i], self.weights[i])
                            for i in range(len(controlpoints))])

    def get_deCasteljauArray(self, t):
        """
        This method calculates and returns a matrix with the lower left corner
        containing the de Casteljau array, calculated for the specified t.
        """
        # initializing the array
        dim = numpy.shape(self.controlpoints)
        deCasteljauArray = scipy.zeros((dim[0],dim[0],dim[1]))
        deCasteljauArray[0] = self.controlpoints

        # filling the array
        for i in range(1, len(deCasteljauArray)):
            for j in range(i, len(deCasteljauArray)):
                deCasteljauArray[i,j] = (
                        (1 - t) * deCasteljauArray[i-1, j-1] +
                        t * deCasteljauArray[i-1,j])
        return deCasteljauArray


if __name__ == '__main__':
    ### Task 4 ###
    controlpoints = scipy.array([[0,0],
                                [4,3],
                                [3,1],
                                [5,1]])
    weights = scipy.array([1,2,5,4])

    # (index of changed weight, the new values)
    changed_weights3 = (2, scipy.array([0,2,3,4,6]))
    changed_weights4 = (-1, scipy.array([0,2,4,6,8]))
    fig, axarr = plt.subplots(2,1,sharex='col')
    for i,plot in enumerate((changed_weights3,changed_weights4)):
        tlist = scipy.linspace(0, 1, 300)
        for new_weight in plot[1]:
            weights[plot[0]] = new_weight
            curve = RationalBezierCurve(controlpoints, weights)
            axarr[i].plot(*zip(*[curve(t)[:2] for t in tlist]), label='Weight w={}'.format(new_weight))

        axarr[i].plot(*zip(*curve.original_controlpoints), 'o--', label='Controlpoints')
        lgd = axarr[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        title = 'Rational Bézier Curve with Control Points $(0,0)$, $(4,3)$, $(3,1)$, $(5,1)$ \n ' \
                'and Varying Weights $[1,2,{},{}]$'.format('w' if i == 0 else 3,
                                                           'w' if i ==1 else 4)
        axarr[i].set_title(title)
    fig.subplots_adjust(hspace=0.5)
    fig.savefig('task4', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
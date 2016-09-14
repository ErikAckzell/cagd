import numpy
import scipy
import pylab


class beziercurve(object):
    def __init__(self, controlpoints):
        self.controlpoints = controlpoints

    def __call__(self, t):
        deCasteljauArray = self.get_deCasteljauArray(t)
        return deCasteljauArray[-1, -2:]

    def subdivision(self, t):
        deCasteljauArray = self.get_deCasteljauArray(t)
        controlpoints1 = scipy.array([deCasteljauArray[i, 2 * i:2 * i+2]
                                      for i in range(len(self.controlpoints))])
        controlpoints2 = scipy.array([deCasteljauArray[-1, 2 * i:2 * i+2]
                                      for i in range(len(self.controlpoints))])[::-1]
        curve1 = beziercurve(controlpoints1)
        curve2 = beziercurve(controlpoints2)

        return (curve1, curve2)

    def get_deCasteljauArray(self, t):
        deCasteljauArray = scipy.column_stack((numpy.copy(self.controlpoints),
                                scipy.zeros((len(self.controlpoints),
                                2 * len(self.controlpoints) - 2))))
        for i in range(1, len(deCasteljauArray)):
            for j in range(1, i + 1):
                deCasteljauArray[i, j*2:j*2+2] = (
                        (1 - t) * deCasteljauArray[i-1, (j-1)*2:(j-1)*2+2] +
                        t * deCasteljauArray[i, (j-1)*2:(j-1)*2+2])
        return deCasteljauArray

    def degree_elevation(self, increase):
        pass

    def plot(self,
             label=None, pointlabel=None, points=300, controlpoints=True):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the controlpoints as well
        """
        # list of u values for which to plot
        tlist = scipy.linspace(0, 1, points)
        pylab.plot(*zip(*[self(t) for t in tlist]), label=label)
        title = 'Bézier curves'
        if controlpoints:  # checking whether to plot control points
            pylab.plot(*zip(*self.controlpoints), 'o--', label=pointlabel)
            title += ' and their control points'
        pylab.legend()
        pylab.title(title)

#controlpoints = scipy.array([[40, 17],
#                             [20, 0],
#                             [18, 8],
#                             [57, -27],
#                             [8, -77],
#                             [-23, -65],
#                             [-100, -15],
#                             [-23, 7],
#                             [-40, 20],
#                             [-15, 10]])

#controlpoints = scipy.array([[x * scipy.cos(x), x * scipy.sin(x)]
#                            for x in scipy.linspace(0, 8 * scipy.pi, 35)])

#controlpoints = scipy.array([[scipy.cos(x), scipy.sin(x)]
#                            for x in scipy.linspace(0, 2 * scipy.pi, 10)])
controlpoints = scipy.array([[-1, 0],
                             [0, 1],
                             [2, 0]])

curve = beziercurve(controlpoints=controlpoints)

curve1, curve2 = curve.subdivision(0.4)

curve1.plot(label='curve1', pointlabel='points1')
curve2.plot(label='curve2', pointlabel='points2')

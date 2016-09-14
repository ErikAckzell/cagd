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

    def get_rectangle(self):
        xlow = min(self.controlpoints[:, 0])
        xhigh = max(self.controlpoints[:, 0])
        ylow = min(self.controlpoints[:, 1])
        yhigh = max(self.controlpoints[:, 1])
        return scipy.array([[xlow, ylow],
                           [xlow, yhigh],
                           [xhigh, yhigh],
                           [xhigh, ylow]])

    def line_functions_from_points(self, p, q):
        Lxcoeff = scipy.polyfit([p[0], q[0]], [p[1], q[1]], 1)
        Lycoeff = scipy.polyfit([p[1], q[1]], [p[0], q[0]], 1)

        def Lx(x):
            return Lxcoeff[0] * x + Lxcoeff[1]

        def Ly(y):
            return Lycoeff[0] * y + Lycoeff[1]

        return Lx, Ly

    def line_crosses_line_segment(self, linefunctions, segmentpoints):
        if segmentpoints[0, 1] == segmentpoints[1, 1]:
            Lx = linefunctions[0]
            if (
                  (Lx(segmentpoints[0, 0]) - segmentpoints[0, 0]) *
                  (Lx(segmentpoints[1, 0]) - segmentpoints[0, 0])) <= 0:
                return True
            else:
                return False
        else:
            Ly = linefunctions[1]
            if (
                  (Ly(segmentpoints[0, 1]) - segmentpoints[0, 0]) *
                  (Ly(segmentpoints[1, 1]) - segmentpoints[0, 0])) <= 0:
                return True
            else:
                return False

    def line_intersects_rectangle(self, rectangle, linefunctions):
        result = False
        for i in range(3):
            segmentpoints = rectangle[i:i+2]
            if self.line_crosses_line_segment(linefunctions=linefunctions,
                                              segmentpoints=segmentpoints):
                result = True
        segmentpoints = scipy.array([rectangle[0], rectangle[3]])
        if self.line_crosses_line_segment(linefunctions=linefunctions,
                                          segmentpoints=segmentpoints):
            result = True
        return result

    def plot_rectangle(self):
        rectangle = self.get_rectangle()
        rectangle_update = scipy.vstack((rectangle, rectangle[0]))
        pylab.plot(rectangle_update[:, 0], rectangle_update[:, 1])

    def plot_line(self, linefunctions):
        Lx = linefunctions[0]
        xmin = min(self.controlpoints[:, 0])
        xmax = max(self.controlpoints[:, 0])
        xlist = scipy.linspace(xmin, xmax, 200)
        ylist = [Lx(x) for x in xlist]
        pylab.plot(xlist, ylist)

    def degree_elevation(self):
        n = len(self.controlpoints)
        new_controlpoints = scipy.zeros((n + 1, 2))
        new_controlpoints[0] = numpy.copy(self.controlpoints[0])
        new_controlpoints[-1] = numpy.copy(self.controlpoints[-1])
        for i in range(1, n):
            new_controlpoints[i] = (
                        (1 - i/n) * numpy.copy(self.controlpoints[i]) +
                        (i/n) * numpy.copy(self.controlpoints[i - 1])
                                   )
        return beziercurve(new_controlpoints)

    def plot(self, label=None, pointlabel=None, points=300, controlpoints=True,
             title=None):
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

#### SUBDIVISION ###
#controlpoints = scipy.array([[-1, 0],
#                             [0, 1],
#                             [2, 0]])

#curve = beziercurve(controlpoints=controlpoints)

#curve1, curve2 = curve.subdivision(0.4)

#curve1.plot(label='curve1', pointlabel='points1')
#curve2.plot(label='curve2', pointlabel='points2')

### DEGREE ELEVATION ###
#curve.plot()
#curve3 = curve.degree_elevation().degree_elevation()
#curve3.plot()
#curve3.plot_rectangle()

### INTERSECTIONS ###
controlpoints = scipy.array([[0, 0],
                             [9, -4],
                             [7, 5],
                             [2, -4]])

curve = beziercurve(controlpoints=controlpoints)
rectangle = curve.get_rectangle()
p = scipy.array([4, 5])
q = scipy.array([6, -4])
linefunctions = curve.line_functions_from_points(p, q)
print(curve.line_intersects_rectangle(rectangle=rectangle,
                                      linefunctions=linefunctions))
curve.plot_rectangle()
curve.plot_line(linefunctions=linefunctions)
pylab.grid()

rectangle = scipy.array([[1, 5],
                         [1, 10],
                         [2, 10],
                         [2, 5]])

print(curve.line_intersects_rectangle(rectangle=rectangle,
                                      linefunctions=linefunctions))

rectangle = scipy.array([[1, 15],
                         [1, 20],
                         [2, 20],
                         [2, 15]])

print(curve.line_intersects_rectangle(rectangle=rectangle,
                                      linefunctions=linefunctions))

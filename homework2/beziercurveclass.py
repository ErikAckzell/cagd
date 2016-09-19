"""
This program consists of three different class definitions, a line class,
a rectangle class and a Bézier curve class. The line and rectangle classes
are used to implement the trivial reject method to determine the intersection
between a Bézier curve and a line.
"""
import numpy
import scipy
import pylab


class rectangle(object):
    """
    This is a rectangle class.
    """
    def __init__(self, xlow, xhigh, ylow, yhigh):
        """
        An object of the class is initialized with two x-values, one for the
        lower bound of the rectangle and one for the upper bound, as well as
        two y-values, one for the lower bound of the rectangle and one for the
        upper bound.
        """
        self.corners = scipy.array([[xlow, ylow],
                                    [xlow, yhigh],
                                    [xhigh, yhigh],
                                    [xhigh, ylow]])
        self.xlow = xlow
        self.xhigh = xhigh
        self.ylow = ylow
        self.yhigh = yhigh

    def plot(self):
        """
        This method plots the rectangle.
        """
        rectangle_update = scipy.vstack((self.corners, self.corners[0]))
        pylab.plot(rectangle_update[:, 0], rectangle_update[:, 1])

    def get_diagonal_length(self):
        """
        This method calculates and returns the length of the diagonal of the
        rectangle.
        """
        return scipy.linalg.norm(self.corners[0] - self.corners[2], 2)

    def get_center(self):
        """
        This method calculates and returns the center of the rectangle.
        """
        xval = 0.5 * (self.xlow + self.xhigh)
        yval = 0.5 * (self.ylow + self.yhigh)
        return scipy.array([xval, yval])


class line(object):
    """
    This is a line class.
    """
    def __init__(self, p, q):
        """
        An object of the class is initialized with two points through which
        the line passes.
        """
        self.Lx, self.Ly = self.get_functions_from_points(p, q)

    def get_functions_from_points(self, p, q):
        """
        This method returns two functions #TODO
        """
        Lxcoeff = scipy.polyfit([p[0], q[0]], [p[1], q[1]], 1)
        Lycoeff = scipy.polyfit([p[1], q[1]], [p[0], q[0]], 1)

        def Lx(x):
            return Lxcoeff[0] * x + Lxcoeff[1]

        def Ly(y):
            return Lycoeff[0] * y + Lycoeff[1]

        return Lx, Ly

    def crosses_line_segment(self, segmentpoints):
        if segmentpoints[0, 1] == segmentpoints[1, 1]:
            if (
                  (self.Lx(segmentpoints[0, 0]) - segmentpoints[0, 1]) *
                  (self.Lx(segmentpoints[1, 0]) - segmentpoints[0, 1])) <= 0:
                return True
            else:
                return False
        else:
            if (
                  (self.Ly(segmentpoints[0, 1]) - segmentpoints[0, 0]) *
                  (self.Ly(segmentpoints[1, 1]) - segmentpoints[0, 0])) <= 0:
                return True
            else:
                return False

    def intersects_rectangle(self, rectangle):
        result = False
        for i in range(3):
            segmentpoints = rectangle.corners[i:i+2]
            if self.crosses_line_segment(segmentpoints=segmentpoints):
                result = True
        segmentpoints = scipy.array([rectangle.corners[0],
                                     rectangle.corners[3]])
        if self.crosses_line_segment(segmentpoints=segmentpoints):
            result = True
        return result

    def plot(self, xmin, xmax):
        xlist = scipy.linspace(xmin, xmax, 200)
        ylist = [self.Lx(x) for x in xlist]
        pylab.plot(xlist, ylist)


class beziercurve(object):
    def __init__(self, controlpoints):
        self.controlpoints = controlpoints
        self.xlow = min(self.controlpoints[:, 0])
        self.xhigh = max(self.controlpoints[:, 0])
        self.ylow = min(self.controlpoints[:, 1])
        self.yhigh = max(self.controlpoints[:, 1])

    def __call__(self, t):
        deCasteljauArray = self.get_deCasteljauArray(t)
        return deCasteljauArray[-1, -2:]

    def subdivision(self, t):
        deCasteljauArray = self.get_deCasteljauArray(t)
        controlpoints1 = scipy.array([deCasteljauArray[i, 2 * i:2 * i+2]
                                      for i in range(len(self.controlpoints))])
        controlpoints2 = scipy.array([deCasteljauArray[-1, 2 * i:2 * i+2]
                                      for i in range(len(self.controlpoints))])
        controlpoints2 = controlpoints2[::-1]
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

    def intersects_line(self, line1):
        rectangle1 = rectangle(xlow=self.xlow,
                               xhigh=self.xhigh,
                               ylow=self.ylow,
                               yhigh=self.yhigh)
        if not line1.intersects_rectangle(rectangle1):
            return False
        else:
            rectangle_list = [rectangle1]
            intersection_list = []
            curve_list = [self]
            while rectangle_list:
                updated_curve_list = []
                updated_rectangle_list = []
                for C in curve_list:
                    C1, C2 = C.subdivision(t=0.5)
                    R1 = rectangle(xlow=C1.xlow,
                                   xhigh=C1.xhigh,
                                   ylow=C1.ylow,
                                   yhigh=C1.yhigh)
                    R2 = rectangle(xlow=C2.xlow,
                                   xhigh=C2.xhigh,
                                   ylow=C2.ylow,
                                   yhigh=C2.yhigh)
                    for RC in [(R1, C1), (R2, C2)]:
                        if line1.intersects_rectangle(RC[0]):
                            updated_rectangle_list.append(RC[0])
                            updated_curve_list.append(RC[1])
                curve_list = updated_curve_list
                rectangle_list = updated_rectangle_list
                poplist = []
                for i, R in enumerate(rectangle_list):
                    if R.get_diagonal_length() < 1e-7:
                        intersection_list.append(R.get_center())
                        poplist.append(i)
                for i in poplist[::-1]:
                    rectangle_list.pop(i)
                    curve_list.pop(i)
            poplist = []
            for i, I in enumerate(intersection_list[:-2]):
                for j, I2 in enumerate(intersection_list[i + 1:]):
                    if scipy.linalg.norm(I - I2, 2) < 1e-7:
                        poplist.append(j + i)
            for i in poplist[::-1]:
                intersection_list.pop(i)
            return intersection_list

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

    def plot(self, controlpoints=True):
        """
        Method to plot the spline.
        points (int): number of points to use when plotting the spline
        controlpoints (bool): if True, plots the controlpoints as well
        """
        # list of u values for which to plot
        tlist = scipy.linspace(0, 1, 300)
        pylab.plot(*zip(*[self(t) for t in tlist]), label='Bézier curve')
        title = 'Bézier curves'
        if controlpoints:  # checking whether to plot control points
            pylab.plot(*zip(*self.controlpoints), 'o--', label='Controlpoints')
            title += ' and their control points'
        pylab.legend()
        pylab.title(title)

#### SUBDIVISION TASSK ###
#controlpoints = scipy.array([[-1, 0],
#                             [0, 1],
#                             [2, 0]])
#
#curve = beziercurve(controlpoints=controlpoints)
#
#curve.plot()
#pylab.grid()
#curve1, curve2 = curve.subdivision(0.4)
#
#print(curve1.controlpoints)
#print(curve2.controlpoints)

#curve1.plot(label='curve1', pointlabel='points1')
#curve2.plot(label='curve2', pointlabel='points2')

### DEGREE ELEVATION ###
#curve.plot()
#curve3 = curve.degree_elevation().degree_elevation()
#print(curve3.controlpoints)
#curve3.plot()
#curve3.plot_rectangle()

### INTERSECTIONS ###
#controlpoints = scipy.array([[0, 0],
#                             [9, -4],
#                             [7, 5],
#                             [2, -4]])
#p = scipy.array([4, 5])
#q = scipy.array([6, -4])
#L = line(p=p, q=q)
#controlpoints = scipy.array([[0, 0],
#                             [9, -4],
#                             [7, 5],
#                             [2, -4]])
#curve = beziercurve(controlpoints=controlpoints)
#print(curve.intersects_line(L))

#curve = beziercurve(controlpoints=controlpoints)
#my_rectangle = curve.get_rectangle()
#p = scipy.array([4, 5])
#q = scipy.array([6, -4])
#linefunctions = curve.line_functions_from_points(p, q)
#print(curve.line_intersects_rectangle(rectangle=my_rectangle,
#                                      linefunctions=linefunctions))
#curve.plot_rectangle()
#curve.plot_line(linefunctions=linefunctions)
#pylab.grid()
#
#my_rectangle = scipy.array([[1, 5],
#                            [1, 10],
#                            [2, 10],
#                            [2, 5]])
#
#print(curve.line_intersects_rectangle(rectangle=my_rectangle,
#                                      linefunctions=linefunctions))
#
#my_rectangle = scipy.array([[1, 15],
#                            [1, 20],
#                            [2, 20],
#                            [2, 15]])
#
#print(curve.line_intersects_rectangle(rectangle=my_rectangle,
#                                      linefunctions=linefunctions))

#curve.plot()
#L.plot(xmin=curve.xlow - 1, xmax=curve.xhigh + 1)
#pylab.grid()
#pylab.show()
#my_rectangle = rectangle(xlow=0,
#                         xhigh=1,
#                         ylow=0,
#                         yhigh=1)
#my_rectangle.plot()
#pylab.plot(my_rectangle.get_center()[0], my_rectangle.get_center()[1], 'bo')
#R1 = rectangle(xlow=1, xhigh=2, ylow=5, yhigh=10)
#print(L.intersects_rectangle(R1))  # True
#R1.plot()
#L.plot(xmin=0, xmax=3)
#pylab.grid()
#pylab.show()
#
#pylab.cla()
#R2 = rectangle(xlow=1, xhigh=2, ylow=15, yhigh=20)
#print(L.intersects_rectangle(R2))  # False
#R2.plot()
#L.plot(xmin=0, xmax=3)
#pylab.grid()
#pylab.show()
#
#R3, R4 = R2.split()
#L.plot(xmin=0, xmax=3)
##R3.plot()
#R4.plot()
#pylab.grid()
#


import scipy
import pylab


class bspline:

    def __init__(self, knots, degree=None, controlpoints=None):
        self.knots = knots

        if controlpoints is not None:
            self.controlpoints = controlpoints
            self.degree = len(self.knots) - 1 - len(self.controlpoints)
            if degree:
                assert degree == self.degree, \
                       'Given degree is wrong. Check the knots and control' + \
                       'points or do not define a degree yourself.'
        else:
            self.degree = degree

    def __call__(self, u):
        """
        Evaluates the spline at a point u, using the spline definition.
        """
        S = sum([controlpoints[i] * self.get_basisfunc(k=self.degree,
                                                       j=i)(u)
                 for i in range(len(controlpoints))])
        return S

    def has_full_support(self, u):
        """
        This method checks if the point u is in an interval with full support.
        """
        if min(scipy.count_nonzero(self.knots < u),
               scipy.count_nonzero(self.knots > u)) > self.degree:
            return True
        else:
            return False

    def get_basisfunc(self, k, j):
        """
        Method that returns a function which evaluates the basis function of
        degree k with index j at point u.
        """
        def basisfunction(u, k=k, j=j):
            """
            Method to evaluate the the basis function N^k with index j at
            point u.
            u (float): the point where to evaluate the basis function
            k (int): the degree of the basis function
            j (int): the index of the basis function we want to evaluate
            knots (array): knot sequence u_i, where i=0,...,K
            """
            if k == 0:
                return 1 if self.knots[j] <= u < self.knots[j+1] \
                         else 0
            else:
                try:
                    a0 = 0 if self.knots[j+k] == self.knots[j] \
                           else (u - self.knots[j]) / (self.knots[j+k] -
                                                       self.knots[j])
                    a1 = 0 if self.knots[j+k+1] == self.knots[j+1] \
                           else (self.knots[j+k+1] - u) / (self.knots[j+k+1] -
                                                           self.knots[j+1])
                    basisfunc = a0 * basisfunction(u, k=k-1, j=j) + \
                                a1 * basisfunction(u, k=k-1, j=j+1)
                except IndexError:
                    numBasisfunc = len(self.knots) - 1 - k
                    raise IndexError('Invalid index. There are no more than {} basis functions for the given problem, choose an ' \
                                'index lower than the number of basis functions.'.format(numBasisfunc))

                return basisfunc
        return basisfunction

    def plot(self):
        """
        This method plots the spline.
        """
        ulist = scipy.linspace(self.knots[0], self.knots[-1], 1000)
        ulist = [u for u in ulist if self.has_full_support(u=u)]
        pylab.plot(*zip(*[self(u=u) for u in ulist]))
        pylab.plot(*zip(*self.controlpoints), 'o--', label='control points')
        pylab.plot(*zip(*[self(u=u) for u in self.knots]), 'rx', label='knots')
        pylab.legend()
        pylab.grid()
        pylab.title('B-spline curve and its control polygon')
        pylab.show()

### Task 2 inspiration###
#knots = scipy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#s = bspline(degree=2, knots=knots)
#ulist = scipy.linspace(s.knots[0], s.knots[-1], 200)
#plotlist = [s.get_basisfunc(k=3, j=4)(u) for u in ulist]
#pylab.plot(ulist, plotlist)


### Task 3 ###
knots1 = scipy.array([0, 0, 1, 1])
knots2 = scipy.array([0, 0, 0, 1, 1, 1])
knots3 = scipy.array([0, 0, 0, 0.3, 0.5, 0.6, 1, 1, 1])

degree = 2

s1 = bspline(knots=knots1, degree=degree)
s2 = bspline(knots=knots2, degree=degree)
s3 = bspline(knots=knots3, degree=degree)

tlist = scipy.array([0.12, 0.24, 0.4, 0.53, 0.78, 0.8])

for i, spline in enumerate([s1, s2, s3]):
    print('For spline {}'.format(i+1))
    for t in tlist:
        print('t={0} : {1}'.format(t, spline.has_full_support(t)))

### Task 4 ###
controlpoints = scipy.array([[0, 0],
                             [3, 4],
                             [7, 5],
                             [9, 2],
                             [13, 1],
                             [10, -1],
                             [7, -1]])

knots = scipy.array([0, 0, 0, 0.3, 0.5, 0.5, 0.6, 1, 1, 1])
s = bspline(knots=knots, controlpoints=controlpoints)
s.plot()

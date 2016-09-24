
import scipy
import pylab


class bspline(object):

    def __init__(self, degree, knots, controlpoints=None):
        self.knots = knots
        self.degree = degree
        self.controlpoints = controlpoints

    def __call__(self, u, degree, controlpoints):
        S = sum([controlpoints[i] * self.get_basisfunc(k=degree,
                                                       j=i)(u)
                 for i in range(len(controlpoints))])
        return S

    def has_full_support(self, u):
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
                           else (u - self.knots[j])/(self.knots[j+k]-self.knots[j])
                    a1 = 0 if self.knots[j+k+1] == self.knots[j+1] \
                           else (self.knots[j+k+1] - u)/(self.knots[j+k+1] - self.knots[j+1])
                    basisfunc = a0 * basisfunction(u, k=k-1) \
                                + a1 * basisfunction(u, k=k-1, j=j+1)
                except IndexError:
                    numBasisfunc = len(self.knots) - 1 - k
                    return 'Invalid index. There are no more than {} basis functions for the given problem, choose an ' \
                                'index lower than the number of basis functions.'.format(numBasisfunc)

                return basisfunc
        return basisfunction

    def plot(self, degree):
        ulist = scipy.linspace(self.knots[0], self.knots[-1], 200)
        ulist = [u for u in ulist if self.has_full_support(u=u, degree=degree)]
        print(ulist)
        pylab.plot(*zip(*[self(u=u, degree=degree) for u in ulist]))
        pylab.plot(*zip(*self.controlpoints), 'o--')
        pylab.show()

### Task 3 ###
#knots1 = scipy.array([0, 0, 1, 1])
#knots2 = scipy.array([0, 0, 0, 1, 1, 1])
#knots3 = scipy.array([0, 0, 0, 0.3, 0.5, 0.6, 1, 1, 1])
#
#degree = 2
#
#s1 = bspline(degree, knots=knots1)
#s2 = bspline(degree=degree, knots=knots2)
#s3 = bspline(degree=degree, knots=knots3)
#
#tlist = scipy.array([0.12, 0.24, 0.4, 0.53, 0.78, 0.8])
#
#for i, spline in enumerate([s1, s2, s3]):
#    print('For spline {}'.format(i+1))
#    for t in tlist:
#        print('t={0} : {1}'.format(t, spline.has_full_support(t)))

### Task 4 ###
controlpoints = scipy.array([[0, 0],
                             [3, 4],
                             [7, 5],
                             [9, 2],
                             [13, 1],
                             [10, -1],
                             [7, -1]])


#ulist = scipy.linspace(s3.knots[0], s3.knots[-1], 200)
#plotlist = [s3.get_basisfunc(k=2, j=1)(u) for u in ulist]
#pylab.plot(ulist, plotlist)

#knots = scipy.array([0, 0, 0, 0.3, 0.5, 0.5, 0.6, 1, 1, 1])
knots = scipy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
s = bspline(degree=2, knots=knots)
ulist = scipy.linspace(s.knots[0], s.knots[-1], 200)
plotlist = [s.get_basisfunc(k=2, j=3)(u) for u in ulist]
pylab.plot(ulist, plotlist)

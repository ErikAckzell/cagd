import scipy
from matplotlib import pyplot as plt
import numpy as np


class Bspline:
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
        S = sum([self.controlpoints[i] * self.get_basisfunc(k=self.degree,j=i)(u)
                 for i in range(len(self.controlpoints))])
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
                if self.knots[j] <= u <= self.knots[j+1]:
                    return 1
                else:
                    return 0
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

    def plot(self, points=None):
        """
        This method plots the spline.
        """
        ulist = scipy.linspace(self.knots[0], self.knots[-1], 1000)
        ulist = [u for u in ulist if self.has_full_support(u=u)]
        plt.plot(*zip(*[self(u=u) for u in ulist]))
        plt.plot(*zip(*self.controlpoints), 'o--', label='control points')
        plt.plot(*zip(*[self(u=u) for u in self.knots]), 'x', label='knots')
        if points is not None:
            plt.plot(*zip(*points), 'o', label='Interpolation Points')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title('B-spline curve and its control polygon')
        plt.show()

class interpolation:
    def __init__(self, points, domain):
        self.points = points
        self.a = domain[0]
        self.b = domain[1]
        self.s = len(points) - 1

    def get_parameter_values(self, method):
        if method == 'uniform':
            return self.uniform_method()
        elif method == 'chord':
            return self.chord_method()
        elif method == 'centripetal':
            return self.centripetal()

    def uniform_method(self):
        t = scipy.array([])
        for k in range(self.s + 1):
            if k == 0:
                val = self.a
            elif k == self.s:
                val = self.b
            else:
                val = self.a + k * (self.b - self.a)/self.s
            t = np.append(t,val)
        return t

    def chord_method(self):
        t = scipy.array([])
        for k in range(self.s + 1):
            if k == 0:
                val = self.a
            elif k == self.s:
                val = self.b
            else:
                Lk = sum([np.linalg.norm(self.points[i] - self.points[i - 1]) for i in range(1,k+1)]) / \
                    sum([np.linalg.norm(self.points[i] - self.points[i - 1]) for i in range(1,self.s+1)])
                val = self.a + Lk * (self.b - self.a)
            t = np.append(t,val)
        return t

    def centripetal(self):
        t = scipy.array([])
        for k in range(self.s + 1):
            if k == 0:
                val = self.a
            elif k == self.s:
                val = self.b
            else:
                Lk = sum([np.sqrt(np.linalg.norm(self.points[i] - self.points[i - 1])) for i in range(1,k+1)]) / \
                    sum([np.sqrt(np.linalg.norm(self.points[i] - self.points[i - 1])) for i in range(1,self.s+1)])
                val = self.a + Lk * (self.b - self.a)
            t = np.append(t,val)
        return t

    def get_knots(self, degree, parameters):
        numKnots = degree + self.s + 2
        knots = scipy.array([])
        for j in range(numKnots):
            if j <= degree:
                knots = np.append(knots, 0)
            elif j >= self.s + 1:
                knots = np.append(knots, 1)
            else:
                jj = j - degree
                u = 1/degree * sum([parameters[i] for i in range(jj,jj + degree - 1 + 1)])
                knots = np.append(knots, u)
        return knots

def get_controlpoints(points, tvals, knots, degree):
    s = len(points) - 1
    N = scipy.zeros((s+1,s+1))
    bspline = Bspline(knots)
    for i in range(s+1): # columns
        basisfunc = bspline.get_basisfunc(degree,i)
        for j in range(s+1): # rows
            N[j,i] = basisfunc(tvals[j])
    print(N)
    controlpoints = np.linalg.lstsq(N,points)[0]
    print(len(controlpoints))
    return controlpoints


if __name__ == '__main__':
    methods = ['uniform', 'chord', 'centripetal']
    points = scipy.array([[0,0],
                          [6,10],
                          [7,10.2],
                          [9,8]])
    degree = [2,3]
    interpolate = interpolation(points, [0, 1])
    fig, axarr = plt.subplots(len(degree),1, figsize=(8,8))
    for i, deg in enumerate(degree):
        for method in methods:
            parameters = interpolate.get_parameter_values(method)
            knots = interpolate.get_knots(deg,parameters)
            controlpoints = get_controlpoints(points,parameters,knots, deg)
            bspline = Bspline(knots,controlpoints=controlpoints)

            ulist = scipy.linspace(knots[0], knots[-1], 300)
            ulist = [u for u in ulist if bspline.has_full_support(u=u)]
            axarr[i].plot(*zip(*[bspline(u=u) for u in ulist]), label=method)
            axarr[i].plot(*zip(*controlpoints), 'o--', label='{} - Control Points'.format(method))
            axarr[i].plot(*zip(*[bspline(u=u+1e-10) for u in knots]), 'x', label='{} - Knots'.format(method))
        axarr[i].plot(*zip(*points), 'o', label='Interpolation Points')
        axarr[i].set_title('B-Spline Curve of Degree {} that Interpolates \n '
                 '$(0,0$, $(6,10)$, $(7,10.2)$ and $(9,8)$'.format(deg))
        lgd = axarr[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        axarr[i].grid()
    fig.tight_layout()
    fig.savefig('task2', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

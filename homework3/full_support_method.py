
import scipy

class bspline(object):

    def __init__(self, knots, degree):
        self.knots = knots
        self.degree = degree

    def has_full_support(self, t):
        if min(scipy.count_nonzero(self.knots < t),
               scipy.count_nonzero(self.knots > t)) > self.degree:
            return True
        else:
            return False

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

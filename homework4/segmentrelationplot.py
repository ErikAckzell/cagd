# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:48:36 2016

@author: erik
"""


from matplotlib import pyplot

pyplot.clf()

nnlist = list(range(2, 9))

for p in range(1, 7):
    ylist = [p * (nn - p) + 1 for nn in nnlist if nn > p]
    pyplot.plot(nnlist[-len(ylist):], ylist, label='p={}'.format(p))

pyplot.xlabel('n + 1')
pyplot.ylabel('total number of controlpoints')
pyplot.legend(loc='best')
#pyplot.title('Relation of B-spline control points and Bézier segments control points')
pyplot.grid()
pyplot.show()

# n + 1 > p

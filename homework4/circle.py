# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:40:04 2016

@author: erik
"""


import scipy
from matplotlib import pyplot


def circle(u):
    return scipy.array([scipy.sin(u), scipy.cos(u)])


def y(t, x):
    return t * (x - 1)

tlist = scipy.linspace(0, 2 * scipy.pi, 200)
xlist = scipy.linspace(-1, 1, 2)
for t in [0, 0.25, 0.5, 1]:
    pyplot.plot(xlist, [y(t, x) for x in xlist])
pyplot.plot(*zip(*[circle(t) for t in tlist]))
pyplot.title('Circle and example lines intersecting (1, 0)\nand lower left circle segment')
pyplot.grid()

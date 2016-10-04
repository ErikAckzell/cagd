# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:20:46 2016

@author: erik
"""


import scipy
from matplotlib import pyplot


def circle1(t):
    return (1 / (1 + t ** 2)) * scipy.array([t ** 2 - 1, - 2 * t])

tlist = scipy.linspace(0, 1, 200)
plotlist1 = [circle1(t) for t in tlist]
pyplot.clf()
pyplot.plot(*zip(*plotlist1), label='circle segment')
pyplot.xlim([-1, 0])
pyplot.ylim([-1, 0])
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.grid()
pyplot.legend(loc='best')

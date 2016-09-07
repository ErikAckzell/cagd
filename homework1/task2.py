# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 07:38:19 2016

This program defines a function to visualize the convex hull of points.
@author: Erik Ackzell
"""

import pylab


def show_hull(points):
    xvals = [pt[0] for pt in points]
    yvals = [pt[1] for pt in points]
    for i in range(len(points) - 1):
        pylab.plot(xvals[i], yvals[i], 'bo')
        for j in range(i + 1, len(points)):
            pylab.plot([xvals[i], xvals[j]], [yvals[i], yvals[j]], 'r')
            print(i, j)

points = [(1, 2), (2, 3), (3, 4), (-1, 2)]
show_hull(points)

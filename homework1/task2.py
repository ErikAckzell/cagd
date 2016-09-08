# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 07:38:19 2016

This program defines a function to visualize the convex hull of points.

@author: Erik Ackzell
"""

import pylab


def show_hull(points):
    """
    This function takes a set of points in the plane and plots all line
    segments connecting the points.

    The boundary of the convex hull is the outermost line segments.
    """
    # Separating the x and y values
    xvals = [pt[0] for pt in points]
    yvals = [pt[1] for pt in points]
    # Nested loop to connect all the points
    for i in range(len(points) - 1):
        # Plotting all the points
        pylab.plot(xvals[i], yvals[i], 'bo')
        for j in range(i + 1, len(points)):
            pylab.plot([xvals[i], xvals[j]], [yvals[i], yvals[j]], 'r')
    # Padding the plot
    padding = 0.1
    pylab.xlim([min(xvals) - padding, max(xvals) + padding])
    pylab.ylim([min(yvals) - padding, max(yvals) + padding])
    pylab.title('Points and their convex hull')
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.show()

# The points to test
points = [(1, 2), (31/6, 3), (-8/3, 4), (0, 5)]
# Calling the function
show_hull(points)

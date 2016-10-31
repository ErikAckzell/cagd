import scipy
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def deCasteljau(n, controlnet, U):
    """
    Uses the de Casteljau Algorithm for triangle patches to evaluate the point on the surface at U.
    :param n: Degree
    :param controlnet: List,
    :param U: point
    :return: Point on the surface
    """
    if len(controlnet) > 1:
        return deCasteljau(n-1, deCasteljauStep(n, controlnet, U), U)
    else:
        return controlnet[0]

def deCasteljauStep(n,controlnet, u):
    """
    Evaluates the new points for the the given control net with the de Casteljau Algorithm for triangle patches.
    :param n: degree
    :param controlnet: list of points, ordered from the top of the triangle going left to right
    :param u: point in 3D
    :return: list with the new control net
    """
    new_controlnet = []
    i, j = 0, 1
    # Iterating over each row in the triangle
    for row in range(1,n+1):
        # Evaluate every new point on that row
        for k in range(row):
            new_point = u[0]*controlnet[i] + u[1]*controlnet[j] + u[2]*controlnet[j+1]
            new_controlnet.append(new_point)
            j += 1
        j += 1
    return new_controlnet

def set_grid(m):
    """
    Creates the grid of U=(u,v,w) where U is a barycentric combination.
    :param m: int, number of points on each axis
    :return: list of tuples (u,v,w)
    """
    return [(i/m,j/m,1-(i+j)/m) for i in range(m,-1,-1) for j in range(m-i, -1, -1)]


def surface_vals(n, grid, controlnet):
    """
    Evaluates the points on the surface. Returns the x, y and z values in separate lists
    :param n: degree
    :param grid: list of points U's
    :param controlnet: list of control points
    :return: x, y and z values in separate lists
    """
    vals = [deCasteljau(n,controlnet, u) for u in grid]
    return zip(*vals)

def plot_surface(n, controlnet, m, title):
    """
    Plots the surface
    :param n: degree
    :param controlnet:
    :param m: number of points to plot
    :param title: Title of the plot
    :return: figure
    """
    grid = set_grid(m)
    x,y,z = surface_vals(n,grid,controlnet)

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x,y,z, cmap=cm.jet, linewidth=0.2)
    ax.set_title(title)
    pyplot.show()
    return fig

if __name__ == '__main__':
    # The controlnet is starting from the top and going from left to right on each row
    controlnetA = scipy.array([(0,6,0),(0,3,0),(3,3,6),(0,0,0),(3,0,0), (6,0,9)])
    controlnetB = scipy.array([(0,6,0),(0,6,6),(3,3,6), (0,3,9),(6,3,15), (6,0,9)])

    n=2
    m=60
    titleA = 'Triangle Patch with Control Net \n $(0,6,0)$, $(0,3,0)$, $(3,3,6)$, $(0,0,0)$, $(3,0,0)$, $(6,0,9)$'
    titleB = 'Triangle Patch with Control Net \n $(0,6,0)$, $(0,6,6)$, $(3,3,6)$, $(0,3,9)$, $(6,3,15)$, $(6,0,9)$'
    #figA = plot_surface(n,controlnetA,m,titleA)
    figB = plot_surface(n,controlnetB,m,titleB)
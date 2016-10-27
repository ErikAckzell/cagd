import scipy.special
import scipy
from matplotlib import pyplot
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D


def B(n, i, t):
    return scipy.special.comb(n, i) * (1 - t)**(n - i) * t**i


def get_2d_grid(datapoints):
    minimum = datapoints.min()
    maximum = datapoints.max()
    grid = (datapoints[:, :2] - minimum) / (maximum - minimum)
    return grid


def evaluate_bezier_surface(controlpoints, u, v, m, n):
    M = scipy.zeros((m + 1) * (n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            M[i * (m + 1) + j] = B(n, i, u) *\
                                 B(n, j, v)
    return M.dot(controlpoints)

if __name__ == '__main__':
    pyplot.close('all')
    datapoints = scipy.array([[-2, -2, 1],
                              [2, 2, 1],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0.5, 0.5, 1]])
    grid = get_2d_grid(datapoints=datapoints)

    m = 1
    n = 1
    K = 4

    M = scipy.zeros((K + 1, (m + 1) * (n + 1)))

    for k in range(K + 1):
        for i in range(m + 1):
            for j in range(n + 1):
                M[k, i * (m + 1) + j] = B(m, i, grid[k, 0]) *\
                                        B(n, j, grid[k, 1])

    A = M.transpose().dot(M)
    b = M.transpose().dot(datapoints)
    controlpoints = scipy.linalg.solve(A, b)

#    degree = 1
    ugrid = scipy.linspace(0, 1, 50)
    vgrid = scipy.copy(ugrid)

    X = scipy.zeros((len(vgrid), len(ugrid)))
    Y = scipy.zeros((len(vgrid), len(ugrid)))
    Z = scipy.zeros((len(vgrid), len(ugrid)))
    for i, v in enumerate(vgrid):
        for j, u in enumerate(ugrid):
            z = evaluate_bezier_surface(controlpoints=controlpoints,
                                        u=u,
                                        v=v,
                                        m=m,
                                        n=n)
            X[i, j] = z[0]
            Y[i, j] = z[1]
            Z[i, j] = z[2]

    figure = pyplot.figure()
    axis = pyplot.subplot(111, projection='3d')
    axis.plot_wireframe(X, Y, Z)
    pyplot.show()

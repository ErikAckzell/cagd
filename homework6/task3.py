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


def evaluate_bezier_surface(controlpoints, u, v, m=1, n=1):
    M = scipy.zeros((m + 1) * (n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            M[i * (m + 1) + j] = B(n, i, u) *\
                                 B(n, j, v)
    return M.dot(controlpoints)


def plane(coeffs, x, y):
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    return scipy.array([x, y, a * x + b * y + c])

if __name__ == '__main__':
    pyplot.close('all')
    datapoints = scipy.array([[-2, -2, 1],
                              [2, 2, 1],
                              [1, 0, 0],
                              [0, 1, 0],
                              [0.5, 0.5, 1]])
    ugrid = scipy.linspace(0, 1, 50)
    vgrid = scipy.copy(ugrid)

    M = scipy.hstack((datapoints[:, :2], scipy.ones((len(datapoints), 1))))
    tmp = datapoints[:, -1]
    A = M.transpose().dot(M)
    b = M.transpose().dot(tmp)

    coeffs = scipy.linalg.solve(A, b)

    xmin = datapoints[:, 0].min()
    xmax = datapoints[:, 0].max()
    ymin = datapoints[:, 1].min()
    ymax = datapoints[:, 1].max()

    controlpoints = scipy.zeros((4, 3))

    controlpoints[0] = plane(coeffs, xmin, ymin)
    controlpoints[1] = plane(coeffs, xmin, ymax)
    controlpoints[2] = plane(coeffs, xmax, ymin)
    controlpoints[3] = plane(coeffs, xmax, ymax)

    X = scipy.zeros((len(vgrid), len(ugrid)))
    Y = scipy.zeros((len(vgrid), len(ugrid)))
    Z = scipy.zeros((len(vgrid), len(ugrid)))
    for i, v in enumerate(vgrid):
        for j, u in enumerate(ugrid):
            z = evaluate_bezier_surface(controlpoints=controlpoints,
                                        u=u,
                                        v=v)
            X[i, j] = z[0]
            Y[i, j] = z[1]
            Z[i, j] = z[2]

    figure = pyplot.figure()
    axis = pyplot.subplot(111, projection='3d')
    axis.plot_wireframe(X, Y, Z)
    axis.scatter(datapoints[:, 0], datapoints[:, 1], datapoints[:, 2],
                 c='r', marker='o', label='data')
    axis.scatter(controlpoints[:, 0], controlpoints[:, 1], controlpoints[:, 2],
                 c='b', marker='d', label='control points')
    axis.legend()
    axis.set_title('Approximating plane to set of data points')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')
    pyplot.show()

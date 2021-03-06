import scipy
from matplotlib import pyplot as plt


class NURBS:
    def __init__(self, grid, controlpoints, degree, weights):
        if len(controlpoints) != len(weights):
            raise ValueError('Not same number of controlpoints and weights')
        try:
            grid = scipy.array(grid)
            grid = grid.reshape((len(grid), 1))
        except ValueError:
            raise ValueError('Grid should be a one-dimensional list or array')
        self.grid = grid.reshape((len(grid), 1))
        self.controlpoints = controlpoints
        self.degree = degree
        self.weights = weights
        self.dim = scipy.shape(controlpoints)[1] + 1

    def __call__(self, u):
        index = self.get_index(u)
        r = self.get_mult(index, u)
        controlpoints, weights = self.get_controlpoints_and_weights(index,
                                                                    r,
                                                                    u)
        current_controlpoints = self.convert_to_homogeneous(
                                                controlpoints=controlpoints,
                                                weights=weights)
        num_controlpoints = len(current_controlpoints)
        d = scipy.zeros((num_controlpoints, num_controlpoints, self.dim))
        d[0] = current_controlpoints
        for s in range(1, num_controlpoints):  # column
            for j in range(s, num_controlpoints):  # rows
                left = index - self.degree + j
                right = index + j - s + 1
                a = (u - self.grid[left])\
                    / (self.grid[right] - self.grid[left])
                d[s, j] = (1 - a) * d[s-1, j-1] + a * d[s-1, j]
        return self.convert_to_cartesian(
            d[-1, -1].reshape((1, len(d[-1, -1])))).flatten()

    def get_mult(self, index, u):
        if u == self.grid[index]:
            return len([i for i in self.grid if i == self.grid[index]])
        elif u == self.grid[-1]:
            return len([i for i in self.grid if i == self.grid[-1]])
        else:
            return 0

    def convert_to_homogeneous(self, controlpoints, weights):
        homogeneous_controlpoints = scipy.zeros((controlpoints.shape[0],
                                                 controlpoints.shape[1] + 1))
        for i in range(len(homogeneous_controlpoints)):
            homogeneous_controlpoints[i] = weights[i] *\
                                           scipy.concatenate((controlpoints[i],
                                                              scipy.ones(1)))
        return homogeneous_controlpoints

    def convert_to_cartesian(self, homogeneous_controlpoints):
        cartesian_controlpoints = scipy.zeros(
            (homogeneous_controlpoints.shape[0],
             homogeneous_controlpoints.shape[1] - 1))

        for i in range(len(cartesian_controlpoints)):
            cartesian_controlpoints[i] = \
                            (1 / homogeneous_controlpoints[i][-1]) *\
                            homogeneous_controlpoints[i][:-1]
        return cartesian_controlpoints

    def get_controlpoints_and_weights(self, index, r, u):
        if r > self.degree:
            if u == self.grid[0]:
                current_controlpoints = self.controlpoints[:self.degree + 1]
                current_weights = self.weights[:self.degree + 1]
            else:
                current_controlpoints = self.controlpoints[-(self.degree + 1):]
                current_weights = self.weights[- (self.degree + 1) :]
        else:
            current_controlpoints = \
                self.controlpoints[index - self.degree:index - r + 1]
            current_weights = self.weights[index - self.degree:index - r + 1]
        return current_controlpoints, current_weights

    def get_index(self, u):
        if u == self.grid[-1]:
            index = (self.grid < u).argmin() - 1
        else:
            index = (self.grid > u).argmax() - 1
        return index

    def plot(self, title, points=500, controlpoints=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ulist = scipy.linspace(self.grid[0], self.grid[-1], points)
        ax.plot(*zip(*[self(u) for u in ulist]), label='NURBS curve')
        if controlpoints:
            ax.plot(*zip(*self.controlpoints), 'o--', label='Control Points')

        lgd = ax.legend(loc='best')
        plt.title(title)
        return fig

if __name__ == '__main__':
    controlpoints = scipy.array([[-1, 0],
                                 [-1, 1],
                                 [0, 1],
                                 [1, 1],
                                 [1, 0],
                                 [1, -1],
                                 [0, -1],
                                 [-1, -1],
                                 [-1, 0]])

    grid = scipy.array([0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1])

    weights = scipy.ones(len(controlpoints))
    for i in [1, 3, 5, 7]:
        weights[i] = (2 ** 0.5) / 2

    nurbscurve = NURBS(controlpoints=controlpoints,
                       degree=2,
                       grid=grid,
                       weights=weights)
    fig = nurbscurve.plot('Circle constructed using NURBS')
    fig.show()

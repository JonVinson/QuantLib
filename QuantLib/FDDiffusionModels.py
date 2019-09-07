import numpy as np

class DiffusionModel:

    def __init__(self, parameters, bounds, n_points):
        self.parameters = parameters
        self.nx = n_points[0]
        a = bounds[0]
        b = bounds[1]
        self.x = np.linspace(a, b, self.nx)
        self.ny = n_points[1]
        a = bounds[2]
        b = bounds[3]
        self.y = np.linspace(a, b, self.ny)

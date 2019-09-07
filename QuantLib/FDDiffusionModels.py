import numpy as np

class DiffusionModel:

    def __init__(self, bounds, n_points):
        self.nx = n_points[0]
        a = bounds[0]
        b = bounds[1]
        self.x = np.linspace(a, b, self.nx)
        self.ny = n_points[1]
        a = bounds[2]
        b = bounds[3]
        self.y = np.linspace(a, b, self.ny)

class SABRModel(DiffusionModel):

    def __init__(self, bounds, n_points):
        super().__init__(bounds, n_points)
        self.F = None
        self.ss = None

    def Calculate(self, p):

        alpha = p[0]
        beta = p[1]
        rho = p[2]
 
        if self.F is None:
            self.F = self.x[:, np.newaxis]

        F = self.F
        sig = self.y
        nx = self.nx
        ny = self.ny

        if self.ss is None:
            self.ss = np.zeros((nx, ny, 2, 2))  

        ss = self.ss
    
        fb = np.power(F, beta)

        ss[:, :, 0, 0] = (sig * fb) ** 2
        ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
        ss[:, :, 0, 1] = ss[:, :, 1, 0]
        ss[:, :, 1, 1] = (alpha * sig) ** 2

        return None, ss

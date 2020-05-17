import numpy as np

class DiffusionModel:

    def __init__(self, bounds = None, n_points = None):
        if bounds is not None and n_points is not None:
            self.SetBounds(bounds, n_points)

    def SetBounds(self, bounds, n_points):
        [self._nx, self._ny] = n_points
        [a, b] = bounds[:2]
        self._x = np.linspace(a, b, self._nx)
        [a, b] = bounds[2:]
        self._y = np.linspace(a, b, self._ny)

class SABRModel(DiffusionModel):

    def __init__(self, bounds = None, n_points = None):
        super().__init__(bounds, n_points)
        self._F = None
        self._ss = None

    def Calculate(self, p):

        [alpha, beta, rho] = p
 
        if self._F is None:
            self._F = self._x[:, np.newaxis]

        F = self._F
        sig = self._y
        nx = self._nx
        ny = self._ny

        if self._ss is None:
            self._ss = np.zeros((nx, ny, 2, 2))  

        ss = self._ss
    
        fb = np.power(F, beta)

        ss[:, :, 0, 0] = (sig * fb) ** 2
        ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
        ss[:, :, 0, 1] = ss[:, :, 1, 0]
        ss[:, :, 1, 1] = (alpha * sig) ** 2

        return None, ss

    def ParameterCount(self):
        return 3
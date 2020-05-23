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

class HestonModel(DiffusionModel):

    def __init__(self, bounds = None, n_points = None):
        super().__init__(bounds, n_points)
        self._S = None
        self._m = None
        self._ss = None

    def Calculate(self, p):

        [mu, theta, kappa, xi, rho] = p
 
        if self._S is None:
            self._S = self._x[:, np.newaxis]

        S = self._S
        nu = self._y
        nx = self._nx
        ny = self._ny

        if self._m is None:
            self._m = np.zeros((nx, ny, 2))  

        m = self._m
    
        m[:, :, 0] = mu * S;
        m[:, :, 1] = kappa * (theta - nu)

        if self._ss is None:
            self._ss = np.zeros((nx, ny, 2, 2))  

        ss = self._ss
    
        ss[:, :, 0, 0] = nu * S ** 2
        ss[:, :, 1, 0] = xi * rho * nu * S;
        ss[:, :, 0, 1] = ss[:, :, 1, 0]
        ss[:, :, 1, 1] = nu * xi ** 2;

        return m, ss

    def ParameterCount(self):
        return 5
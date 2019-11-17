import numpy as np
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d
from FDSolver import FDSolver2D
from FDDiffusionModels import DiffusionModel

class Calibrator:
    
    def __init__(self):
        self.model = None

    def SetModel(self, model):
        self.model = model
        self._setupModel()
        self._setupVariables()

    def SetParameters(self, varParams, pBounds = None, varIndex = None, fixParams = None):
        self.varParams = varParams
        self.pBounds = pBounds
        self.varIndex = varIndex
        self.fixParams = fixParams
        self._setupVariables()

    def SetLattice(self, bounds, n_points, T, n_steps):
        self.bounds = bounds
        self.n = n_points
        self.T = T
        self.nt = n_steps
        self._setupSolver()
        self._setupModel()
        self._setupDiffusion()

    def SetDiffusionStart(self, diffStart):
        self.diffStart = diffStart
        self._setupDiffusion()

    def SetDistribution(knownDist, xDist = None):
        self.knownDist = knownDist
        self.xDist = xDist

    def GetResult(self):
        return _calibrate(self)

    def _setupSolver(self):
        [nx, ny] = self.n
        [a, b] = self.bounds[:2]
        dx = (b - a) / (nx - 1)
        [a, b] = self.bounds[2:]
        dy = (b - a) / (ny - 1)
        dt = self.T / self.nt
        self.solver = FDSolver2D(nx, dx, ny, dy, self.nt, dt)
        self.dx = dx
        self.dy = dy

    def _setupModel(self):
        if self.model is not None and self.bounds is not None:
            self.model.setBounds(self.bounds, self.n)

    def _setupVariables(self):
        if self.model is not None and self.varParams is not None:
            self.n_par = self.model.ParameterCount()
            self.n_var = len(self.varParams)
            self.n_fix = 0 if self.fixParams is None else len(self.fixParams)
            if (self.n_par != self.n_var + self.n_fix):
                print("Model requires " + self.n_par + " parameters.")
            if self.varIndex is None:
                self.varIndex = range(self.n_par)
            self.fixedIndex = np.setdiff1d(range(self.n_par), self.varIndex)
    
    def _setupDiffusion(self):
        if self.bounds is not None and self.diffStart is not None:
            self.ix = int((self.diffStart[1] - self.bounds[0]) / self.dx)
            self.iy = int((self.diffStart[3] - self.bounds[2]) / self.dy)

    def _calibrate(self):

        F = np.linspace(self.bounds[0], self.bounds[1], self.n[0])

        solver = self.solver
        solver.cond = np.empty(self.n)

        if self.xDist is None:
            self.xDist = F

        q = np.zeros(self.n_par)

        def opt_fun(p):
            for i in range(self.n_var):
                q[self.varIndex[i]] = p[i]
            for i in range(self.n_fix):
                q[self.fixedIndex[i]] = self.fixParams[i]
            [solver.mu, solver.ss] = self.model.Calculate(q)
            solver.cond[:] = 0
            solver.cond[self.ix, self.iy] = 1
            solver.a[:] = 0
            solver.SolveForward()
            dist2d = solver.Solution()
            fintp = interp1d(F, np.sum(dist2d[-1], axis=1), kind='cubic')
            G = fintp(self.xDist)
            dist = G / np.sum(G)
            return -np.sum(self.knownDist * np.log(dist))

        return minimize(opt_fun, self.varParams, bounds=self.pBounds)
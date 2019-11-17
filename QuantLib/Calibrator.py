import numpy as np
from scipy.optimize import least_squares, minimize, brute
from scipy._interpolate import interp1d
from FDSolver import FDSolver2D
from FDDiffusionModels import DiffusionModel

class Calibrator:
    
    def __init__(self):
        self._model = None
        self._bounds = None
        self._varParams = None
        self._diffStart = None
        self._modelSetup = False
        self._variableSetup = False
        self._diffusionSetup = False
        self._distributionSetup = False

    def SetModel(self, model):
        self._model = model
        self._setupModel()
        self._setupVariables()

    def SetParameters(self, varParams, pBounds = None, varIndex = None, fixParams = None):
        self._varParams = varParams
        self._pBounds = pBounds
        self._varIndex = varIndex
        self._fixParams = fixParams
        self._setupVariables()

    def SetLattice(self, bounds, n_points, T, n_steps):
        self._bounds = bounds
        self._n = n_points
        self._T = T
        self._nt = n_steps
        self._setupSolver()
        self._setupModel()
        self._setupDiffusion()

    def SetDiffusionStart(self, diffStart):
        self._diffStart = diffStart
        self._setupDiffusion()

    def SetDistribution(self, knownDist, xDist = None):
        self._knownDist = knownDist
        self._xDist = xDist
        self._distributionSetup = True

    def GetResult(self):
        return _calibrate(self)

    def _setupSolver(self):
        [nx, ny] = self._n
        [a, b] = self._bounds[:2]
        dx = (b - a) / (nx - 1)
        [a, b] = self._bounds[2:]
        dy = (b - a) / (ny - 1)
        dt = self._T / self._nt
        self._solver = FDSolver2D(nx, dx, ny, dy, self._nt, dt)
        self._dx = dx
        self._dy = dy

    def _setupModel(self):
        if self._model is not None and self._bounds is not None:
            self._model.setBounds(self._bounds, self._n)
            self._setupModel = True

    def _setupVariables(self):
        if self._model is not None and self._varParams is not None:
            self._n_par = self._model.ParameterCount()
            self._n_var = len(self._varParams)
            self._n_fix = 0 if self._fixParams is None else len(self._fixParams)
            if (self._n_par != self._n_var + self._n_fix):
                print("Model requires " + self._n_par + " parameters.")
            if self._varIndex is None:
                self._varIndex = range(self._n_par)
            self._fixedIndex = np.setdiff1d(range(self._n_par), self._varIndex)
            self._setupVariables = True
    
    def _setupDiffusion(self):
        if self._bounds is not None and self._diffStart is not None:
            self._ix = int((self._diffStart[1] - self._bounds[0]) / self._dx)
            self._iy = int((self._diffStart[3] - self._bounds[2]) / self._dy)
            self._diffusionSetup = True

    def _validateSetup(self):
        valid = self._modelSetup and self._variableSetup and self._diffusionSetup and self._distributionSetup
        if ~self._modelSetup:
            print("Model not set")
        if ~self._variableSetup:
            print("Variables not initialized")
        if ~self._setupDiffusion:
            print("Diffusion not initialized")
        if ~self._distributionSetup:
            print("Known distribution not set")
        return valid
            
    def _calibrate(self):
        
        if ~self._validateSetup():
            return None

        solver = self._solver
        solver.cond = np.empty(self._n)

        F = np.linspace(self._bounds[0], self._bounds[1], self._n[0])

        if self._xDist is None:
            self._xDist = F

        q = np.zeros(self._n_par)

        def opt_fun(p):
            for i in range(self._n_var):
                q[self._varIndex[i]] = p[i]
            for i in range(self._n_fix):
                q[self._fixedIndex[i]] = self._fixParams[i]
            [solver.mu, solver.ss] = self._model.Calculate(q)
            solver.cond[:] = 0
            solver.cond[self._ix, self._iy] = 1
            solver.a[:] = 0
            solver.SolveForward()
            dist2d = solver.Solution()
            fintp = interp1d(F, np.sum(dist2d[-1], axis=1), kind='cubic')
            G = fintp(self._xDist)
            dist = G / np.sum(G)
            return -np.sum(self._knownDist * np.log(dist))

        return minimize(opt_fun, self._varParams, bounds=self._pBounds)
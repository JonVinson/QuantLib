import numpy as np
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d
from FDSolver import FDSolver2D
from FDDiffusionModels import DiffusionModel

def calibrate(self, model, initParams, pBounds, bounds, n, T, nt, diffStart, knownDist, xDist):

    [nx, ny] = n

    [a, b] = bounds[:2]
    dx = (b - a) / (nx - 1)
    [a, b] = bounds[2:]
    dy = (b - a) / (ny - 1)
    dt = T / nt

    [ix, iy] = diffStart

    solver = FDSolver2D(nx, dx, ny, dy, nt, dt)
    solver.cond = np.empty((nx, ny))

    model.SetBounds(bounds, n)

    knownDist = knownDist[:, np.newaxis]

    def opt_fun(p):
        [solver.mu, solver.ss] = model.Calculate(p)
        solver.cond[:] = 0
        solver.cond[ix, iy] = 1
        solver.a[:] = 0
        solver.SolveForward()
        dist2d = solver.Solution()
        [a, b] = bnds[:1]
        F = np.linspace(a, b, nx)
        fintp = interp1d(F, np.sum(dist2d[-1],axis=1), kind='cubic')
        G = fintp(xDist)
        dist = G / np.sum(G)
        return -np.sum(knownDist * np.log(dist))

    return minimize(opt_fun, initParams, pBounds)
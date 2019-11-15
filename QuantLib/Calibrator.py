import numpy as np
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d
from FDSolver import FDSolver2D
from FDDiffusionModels import DiffusionModel

def calibrate(model, varParams, pBounds, bounds, n, T, nt, diffStart, knownDist, xDist, fixParams = None, varIndex = None):

    [nx, ny] = n

    [a, b] = bounds[:2]
    dx = (b - a) / (nx - 1)
    ix = int((diffStart[0] - a) / dx)

    [a, b] = bounds[2:]
    dy = (b - a) / (ny - 1)
    iy = int((diffStart[1] - a) / dy)

    dt = T / nt

    solver = FDSolver2D(nx, dx, ny, dy, nt, dt)
    solver.cond = np.empty((nx, ny))

    model.SetBounds(bounds, n)

    knownDist = knownDist[:, np.newaxis]

    n_par = model.ParameterCount()
    n_var = len(varParams)
    n_fix = 0 if fixParams is None else len(fixParams)

    if (n_par != n_var + n_fix):
        print("Model requires " + n_par + " parameters.")
        return

    q = np.zeros(n_par)

    if varIndex is None:
        varIndex = range(n_par)

    n_var = len(varIndex) 
    fixIndex = np.setdiff1d(range(n_par), varIndex)

    def opt_fun(p):
        for i in range(n_var):
            q[varIndex[i]] = p[i]
        for i in range(n_fix):
            q[fixIndex[i]] = fixParams[i]
        [solver.mu, solver.ss] = model.Calculate(q)
        solver.cond[:] = 0
        solver.cond[ix, iy] = 1
        solver.a[:] = 0
        solver.SolveForward()
        dist2d = solver.Solution()
        [a, b] = bounds[:2]
        F = np.linspace(a, b, nx)
        fintp = interp1d(F, np.sum(dist2d[-1], axis=1), kind='cubic')
        G = fintp(xDist)
        dist = G / np.sum(G)
        return -np.sum(knownDist * np.log(dist))

    return minimize(opt_fun, varParams, bounds=pBounds)
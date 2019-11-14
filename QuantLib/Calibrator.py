import numpy as np
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d
from FDSolver import FDSolver2D
from FDDiffusionModels import DiffusionModel

def calibrate(model, initParams, pBounds, bounds, n, T, nt, diffStart, knownDist, xDist, pIndex = None):

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

    npar = len(initParams)
    q = np.zeros(npar)

    if pIndex is None:
        pIndex = range(npar)

    nidx = len(pIndex) 
    pComp = set(range(npar)) - set(pIndex)

    def opt_fun(p):
        for i in range(nidx):
            q[pIndex[i]] = p[i]
        for i in range(npar - nidx):
            q[pComp[i]] = initParams[pComp[i]]
        [solver.mu, solver.ss] = model.Calculate(q)
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

    p0 = np.zeros(nidx)
    
    for i in range(nidx):
        p0[i] = initParams[pIndex[i]]

    return minimize(opt_fun, p0, pBounds)
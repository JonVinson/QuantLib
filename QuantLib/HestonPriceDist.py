import numpy as np
from scipy.interpolate import *

from Processes import *
from FDSolver import *
from FDDiffusionModels import *
from FiniteDifference import *

################################################################################################################

def heston_dist_fd2(mu, theta, kappa, xi, rho, sig0, f0, T, bnds, nx, ny, n_steps):

    a = bnds[0]
    b = bnds[1]
    Fa = a
    dx = (b - a) / (nx - 1)

    a = bnds[2]
    b = bnds[3]
    siga = a
    dy = (b - a) / (ny - 1)
    dt = T[-1] / n_steps

    fds = FDSolver2D([nx, ny, n_steps + 1], [dx, dy, dt])
    fds.SetCondition(np.empty((nx, ny)))
    model = HestonModel(bnds, [nx, ny])
        
    coeff = model.Calculate([mu, theta, kappa, xi, rho])
    fds.SetCoefficients(coeff)

    ix = int((f0 - Fa) / dx)
    iy = int((sig0 - siga) / dy)

    fds.Condition()[:] = 0
    fds.Condition()[ix, iy] = 1

    fds.SolveForward()

    return fds.Solution()

##############################################################################################################

def heston_price_dist_fd2(mu, theta, kappa, xi, rho, sig0, f0, K, T, bnds, nx, ny, n_steps):

    g = heston_dist_fd2(mu, theta, kappa, xi, rho, sig0, f0, T, bnds, nx, ny, n_steps)
        
    a = bnds[0]
    b = bnds[1]
    F = np.linspace(a, b, nx)
    FF = np.linspace(a, b, 10 * nx - 9)

    dt = T[-1] / n_steps

    P = np.zeros((len(K), len(T)))

    for j in range(len(T)):
        k = int(np.rint((T[j] / dt)))
        fintp = interp1d(F, np.sum(g[k],axis=1), kind='cubic')
        G = fintp(K)
        P[:, j] = G / np.sum(G)

    return P

##############################################################################################################

def heston_prices_fd2(mu, theta, kappa, xi, rho, sig0, f0, K, T, bnds, nx, ny, n_steps):

    g = heston_dist_fd2(mu, theta, kappa, xi, rho, sig0, f0, T, bnds, nx, ny, n_steps)
        
    a = bnds[0]
    b = bnds[1]
    F = np.linspace(a, b, nx)
    FF = np.linspace(a, b, 10 * nx - 9)

    dt = T[-1] / n_steps

    P = np.zeros((len(K), len(T)))

    for i in range(len(K)):        
        f = lambda x : np.where(x < K[i], 0.0, x - K[i])
        for j in range(len(T)):
            k = int(np.rint((T[j] / dt)))
            fintp = interp1d(F, np.sum(g[k],axis=1), kind='cubic')
            G = fintp(FF)
            P[i, j] = np.sum(G * f(FF)) / np.sum(G)

    return P


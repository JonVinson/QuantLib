import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d

from Processes import *
from BlackScholes import *
from FDPricing import *
from FDSolve import *
from FDSolver import *
from FDDiffusionModels import *
from FiniteDifference import *

################################################################################################################

def sabr_prices_mc(alpha, beta, rho, sig0, f0, mu, K, T, Z1, Z2, dt):

    (S, a, z) = sabr(0, 0, alpha=alpha, beta=beta, rho=rho, sig0=sig0, f0=f0, mu=mu, dt=dt, Z1=Z1, Z2=Z2)

    D = np.exp(-mu * T)

    P = np.zeros((len(K), len(T)))

    for i in range(len(K)):
        
        f = lambda x : np.where(x < K[i], 0.0, x - K[i])

        for j in range(len(T)):

            k = int(np.rint((T[j] / dt)))

            P[i, j] = np.mean(f(S[:, k])) * D[j]

    return P

##############################################################################################################

def sabr_dist_fd2(alpha, beta, rho, sig0, f0, mu, T, bnds, nx, ny, n_steps):

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
    sabr = SABRModel(bnds, [nx, ny])
        
    coeff = sabr.Calculate([alpha, beta, rho])
    fds.SetCoefficients(coeff)

    ix = int((f0 - Fa) / dx)
    iy = int((sig0 - siga) / dy)

    fds.Condition()[:] = 0
    fds.Condition()[ix, iy] = 1

    fds.SolveForward()

    return fds.Solution()

##############################################################################################################

def sabr_price_dist_fd2(alpha, beta, rho, sig0, f0, mu, K, T, bnds, nx, ny, n_steps):

    g = sabr_dist_fd2(alpha, beta, rho, sig0, f0, mu, T, bnds, nx, ny, n_steps)
        
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

def sabr_prices_fd2(alpha, beta, rho, sig0, f0, mu, K, T, bnds, nx, ny, n_steps):

    g = sabr_dist_fd2(alpha, beta, rho, sig0, f0, mu, T, bnds, nx, ny, n_steps)
        
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


import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, minimize, brute
from scipy.interpolate import interp1d

from Processes import *
from BlackScholes import *
from FDPricing import *
from FiniteDifference import *
from Calibrator import calibrate

from FDSolver import FDSolver2D
from FDDiffusionModels import SABRModel

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

fds = None
sabr = None

def sabr_dist_fd2(alpha, beta, rho, sig0, f0, mu, T, bnds, nx, ny, n_steps):

    global fds, sabr

    a = bnds[0]
    b = bnds[1]
    Fa = a
    dx = (b - a) / (nx - 1)

    a = bnds[2]
    b = bnds[3]
    siga = a
    dy = (b - a) / (ny - 1)
    dt = T[-1] / n_steps

    if fds is None:
        fds = FDSolver2D(nx, dx, ny, dy, n_steps, dt)
        fds.cond = np.empty((nx, ny))
        sabr = SABRModel(bnds, [nx, ny])
        
    [mean, ss] = sabr.Calculate([alpha, beta, rho])
    fds.ss = ss

    fds.cond[:] = 0
    ix = int((f0 - Fa) / dx)
    iy = int((sig0 - siga) / dy)
    fds.cond[ix, iy] = 1

    fds.a[:] = 0

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

##############################################################################################################

alpha = 0.2
beta = 1.0
rho = -0.4
sig0 = 0.2

f0 = 1
mu = None

a = 0.25
b = 1.75
K = np.linspace(a, b, 41)

T = 1.0
#t = np.array([1/12, 1/6, 0.25, 1/3, 0.5, 1, 2])
t = np.array([T])

n_steps = 60
nx = 41
ny = 41
bnds = (0, 4, 0, 1)

G0 = sabr_price_dist_fd2(alpha, beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
G0 = G0 / np.sum(G0)
P = sabr_prices_fd2(alpha, beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
G = np.abs(d2(P[:, -1]))
G = G / np.sum(G)

plt.plot(K, G0, '--')
plt.plot(K, G)
plt.show()

pBnds = ((0, 1), (-1, 1))
initParams = [0.5, beta, 0]
pIndex = [0, 2]

result = calibrate(SABRModel(), initParams, pBnds, bnds, [nx, ny], n_steps, T, [f0, sig0], G, K)

print("FD result: ", result)

x = result.x
#D = sabr_price_dist_fd2(x[0], beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)

#plt.plot(K, G, '--', K, D, '-')
#plt.show()
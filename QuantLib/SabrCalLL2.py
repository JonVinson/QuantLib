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
from Calibrator import Calibrator

from FDSolver import FDSolver2D
from FDDiffusionModels import SABRModel

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

pBnds = ((0, 1), (-1, 1))
varParams = (0.5, 0)
varIndex = [0, 2]
fixParams = [beta]
model = SABRModel()

cal = Calibrator()

cal.SetModel(model)
cal.SetParameters(varParams, pBnds, varIndex, fixParams)
cal.SetLattice(bnds, [nx, ny], t, n_steps)
cal.SetDiffusionStart([f0, sig0])
cal.SetDistribution(G, K)

result = cal.GetResult()

print("FD result: ", result)

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

import FDSolver;

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
init = None
ss = None

def sabr_dist_fd2(alpha, beta, rho, sig0, f0, mu, T, bnds, nx, ny, n_steps):

    global fds

    a = bnds[0]
    b = bnds[1]
    F = np.linspace(a, b, nx)
    dx = (b - a) / (nx - 1)

    # underlying vol range
    a = bnds[2]
    b = bnds[3]
    sig = np.linspace(a, b, ny)
    dy = (b - a) / (ny - 1)
    dt = T[-1] / n_steps

    if fds is None:
        fds = FDSolver(nx, dx, ny, dy, n_steps + 1, dt)
        init = np.empty((nx, ny))
        ss = np.zeros((nx, ny, 2, 2))

    fb = np.power(F[:, np.newaxis], beta)
    ss[:, :, 0, 0] = (sig * fb) ** 2
    ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
    ss[:, :, 0, 1] = ss[:, :, 1, 0]
    ss[:, :, 1, 1] = (alpha * sig) ** 2

    init[:] = 0
    ix = int((f0 - F[0]) / dx)
    iy = int((sig0 - sig[0]) / dy)
    init[ix, iy] = 1

    fds.Set(init, mu, ss)
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

#dt = T / n_steps

#np.random.seed(0)
#n_paths = 62500
#Z1 = gaussian(n_paths, n_steps)
#Z2 = gaussian(n_paths, n_steps)
#Z1[np.abs(Z1) > 6.0] = 0.0
#Z2[np.abs(Z2) > 6.0] = 0.0

G0 = sabr_price_dist_fd2(alpha, beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
G0 = G0 / np.sum(G0)
P = sabr_prices_fd2(alpha, beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
G = np.abs(d2(P[:, -1]))
G = G / np.sum(G)
G = G[:, np.newaxis]

plt.plot(K, G0, '--')
plt.plot(K, G)
plt.show()

#P = P * (1 + 0.01 * np.random.randn())
#P = sabr_prices_fd(alpha, beta, rho, sig0, f0, mu, K, t)
#P = sabr_prices_mc(alpha, beta, rho, sig0, f0, mu, K, t, Z1, Z2, dt)

#print(np.transpose(P,[1,0]))

#n_paths = 100000
#n_steps = 16
#dt = T / n_steps

#Z1 = gaussian(n_paths, n_steps)
#Z2 = gaussian(n_paths, n_steps)
#Z1[np.abs(Z1) > 6.0] = 0.0
#Z2[np.abs(Z2) > 6.0] = 0.0

x_interp = np.linspace(bnds[0], bnds[1], nx);

def fun(x):
    D = sabr_price_dist_fd2(x[0], beta, x[1], sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
    D = D / np.sum(D)
    return -np.sum(G * np.log(D))
    
result = minimize(fun, [0.5,0], bounds=((0,1),(-1,1)))

print("FD result: ", result)

x = result.x
#D = sabr_price_dist_fd2(x[0], beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)

#plt.plot(K, G, '--', K, D, '-')
#plt.show()
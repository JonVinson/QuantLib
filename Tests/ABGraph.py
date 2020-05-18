import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from FDSolve import *
from Processes import *

alpha = 0.2
beta = 0.95
rho = 0.0

f0 = 100
sig0 = 0.1

a = 0.0
b = 2.0 * f0
nx = 41
F = np.linspace(a, b, nx)

dx = (b - a) / (nx - 1)

a = 0.0
b = 2.0 * sig0
ny = 41
sig = np.linspace(a, b, ny)

dy = (b - a) / (ny - 1)

T = 2.0    
nt = 160
dt = T / nt

init = np.zeros((nx, ny))
ix = int((f0 - F[0]) / dx)
iy = int((sig0 - sig[0]) / dy)
init[ix, iy] = 1

ss = np.empty((nx, ny, 2, 2))

alphas = np.linspace(0.1, 0.5, 9)
betas = np.linspace(0.5, 1.0, 11)

prices = np.empty((len(alphas), len(betas)))

f = lambda x : np.where(x < f0, 0, x - f0)

for i in range(len(alphas)):
    
    alpha = alphas[i]
    
    for j in range(len(betas)):
    
        beta = betas[j]
        
        fb = np.power(F[:, np.newaxis], beta)
        ss[:, :, 0, 0] = (sig * fb) ** 2
        ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
        ss[:, :, 0, 1] = ss[:, :, 1, 0]
        ss[:, :, 1, 1] = (alpha * sig) ** 2

        g = fd_solve_fwd_2d(init, None, ss, F, sig, nt, dt)
        prices[i, j] = np.sum(np.sum(g[nt], axis=1) * f(F))

[X, Y] = np.meshgrid(betas, alphas)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, prices)
plt.show()

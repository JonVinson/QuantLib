import numpy as np
import matplotlib.pyplot as plt

from FDSolve import *
from Processes import *

alpha = 0.2
beta = 0.95
rho = -0.5

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

ss = np.empty((nx, ny, 2, 2))
fb = np.power(F[:, np.newaxis], beta)
ss[:, :, 0, 0] = (sig * fb) ** 2
ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
ss[:, :, 0, 1] = ss[:, :, 1, 0]
ss[:, :, 1, 1] = (alpha * sig) ** 2

T = 2.0    
nt = 80
dt = T / nt

init = np.zeros((nx, ny))
ix = int((f0 - F[0]) / dx)
iy = int((sig0 - sig[0]) / dy)
init[ix, iy] = 1

g = fd_solve_fwd_2d(init, None, ss, F, sig, nt, dt)

pfd = np.sum(g[nt], axis=1)

n_paths = 100000

S = sabr(n_paths, nt, 0, alpha, beta, rho, f0, sig0, dt)[0]

pmc = np.histogram(S[:, nt], nx, (F[0] - dx / 2, F[-1] + dx / 2))[0] / n_paths

print(np.sum(pfd), np.sum(pmc))

plt.plot(F, pfd, label='FD')
plt.plot(F, pmc, label='MC')
plt.legend()
plt.show()
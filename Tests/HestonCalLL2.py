import numpy as np

from Calibrator import Calibrator
from FDDiffusionModels import HestonModel

from HestonPriceDist import *

##############################################################################################################

mu = 0.01
theta = 0.2
kappa = 0.1
xi = 0.2
rho = -0.4

sig0 = 0.25

f0 = 1

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

G0 = heston_price_dist_fd2(mu, theta, kappa, xi, rho, sig0, f0, K, t, bnds, nx, ny, n_steps)
G0 = G0 / np.sum(G0)
P = heston_prices_fd2(mu, theta, kappa, xi, rho, sig0, f0, K, t, bnds, nx, ny, n_steps)
G = np.abs(d2(P[:, -1]))
G = G / np.sum(G)

pBnds = ((0, 1), (-1, 1))
varParams = (0.5, 0)
varIndex = [3, 4]
fixParams = [mu, theta, kappa]

model = HestonModel()

cal = Calibrator()

cal.SetModel(model)
cal.SetParameters(varParams, pBnds, varIndex, fixParams)
cal.SetLattice(bnds, [nx, ny], t, n_steps)
cal.SetDiffusionStart([f0, sig0])
cal.SetDistribution(G, K)

result = cal.GetResult()

print("FD result: ", result)

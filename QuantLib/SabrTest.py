#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom

from MCPaths import *
from BlackScholes import *

# SABR parameters

sig0 = 0.1
alpha = 0.5
beta = 0.9
rho = -0.75

mu = 0.05
S = 125                          # underlying price
K = np.linspace(100, 150, 11)    # strike price range

T = 1.0 # time to maturity

# Construct underlying paths using SABR model

n_paths = 1000
n_steps = 10

dt = T / n_steps

paths = SABRPaths(n_paths, n_steps, mu, alpha, beta, rho, S, sig0, dt)

x = paths.X[:, -1]

pbs = np.zeros(len(K))
psabr = np.zeros(len(K))
ivol = np.zeros(len(K))

for i in range(len(K)):
    k = K[i]
    f = lambda x : np.where(x < k, 0.0, x - k)
    p = np.mean(f(x))
    ivol[i] = bs_eur_call_ivol(p, k, S, 0.0, 0.0, T, sig0)
    psabr[i] = 0

plt.plot(K, ivol)
plt.show()
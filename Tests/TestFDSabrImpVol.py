##########################################################################################################################
# This program demonstrates the use of the Tsitsiklis-Van Roy Algorithm, the Longstaff-Schwartz algorithm, and a
# finite difference algorithm to estimate the value of an American option for a range of underlying prices.
# The Black-Scholes value of the corresponding European option is also calculated for comparison.
#
# The program depends on the following modules:
#
#   BlackSholes.py  : Black-Scholes prices
#   FDPRicing.py    : Finite difference algorithm
#   MCPricing.py    : TVR, LS, and AB algorithms
#   MCPaths.py      : Classes to contain Monte-Carlo paths and associated parameters
#   Processes.py    : Generate Monte-Carlo paths
#   Regressors.py   : Transformation functions used for parametric regression estimate of conditional expectation values
##########################################################################################################################

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

from BlackScholes import *
from FDPricing import fd_option
from FDPricing0 import fd_option as fd_option0

alpha = 0.2
beta = 0.95
rho = -0.5

T = 1.0     # time to maturity

n_steps = 1000     # time steps
dt = T / n_steps    # time step size

#discount = np.exp(-mu * dt)    # discount factor per time step

# option payoff function

f_call = lambda x : np.where(x < K, 0.0, x - K)
f_put = lambda x : np.where(x < K, K - x, 0.0)

fpay0 = f_call

# underlying price range

K = 100.0   # strike price
a = 0.5 * K
b = 1.5 * K
F = np.linspace(a, b, 41)

# underlying vol range
sig0 = 0.08
a = 0.0
b = 2.0 * sig0
sig = np.linspace(a, b, 41)

# Finite difference algorithm (new)

def fpay(x):
    [F, s] = x
    return np.full((len(F), len(s)), fpay0(F)[:, np.newaxis])

def fmu(x):
    [F, s] = x
    return np.zeros((len(F), len(s), 2))

def fsig(x):
    [F, s] = x
    sig = np.zeros((len(F), len(s), 2, 2))
    sig[:, :, 0, 0] = s * F[:, np.newaxis] ** beta
    sig[:, :, 1, 0] = alpha * rho * s
    sig[:, :, 1, 1] = alpha * np.sqrt(1.0 - rho ** 2) * s
    return sig

def fdisc(x):
    [F, s] = x
    return np.zeros((len(F), len(s)))

S = K
F1 = F[(F >= 0.8 * S) & (F <= 1.2 * S)]
ivol = np.zeros(len(F1))

for i in range(len(ivol)):
    K = F1[i]
    g = fd_option(fpay, fmu, fsig, fdisc, [F, sig], dt, n_steps, am=False)
    fd_price = g[0, 20, 20]
    ivol[i] = bs_eur_call_ivol(fd_price, K, S, 0, 0, T)

# Show results

fig, ax = plt.subplots()

ax.plot(F1, ivol)

plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")

title = "American " + ("Call" if fpay0 == f_call else "Put") + " Option Implied Volatilities by Finite Difference"
plt.title(title)

#offset = OffsetFrom(leg, (0.05, 0.0))
#annotation = ("r = {0:5.2f}\n" + \
#             "q = {1:5.2f}\n" + \
#             "$\sigma$ = {2:5.2}\n" + \
#             "K = {3:5.1f}\n" + \
#             "T = {4:5.2f}").format(mu, div, sig, K, T)
#ax.annotate(annotation, xy=(0,0),
#            xycoords='figure fraction', xytext=(0,-20), textcoords=offset, 
#            horizontalalignment='left', verticalalignment='top')


plt.show()

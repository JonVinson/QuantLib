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
from FDSolve import *

mu = 0.05   # interest rate
div = 0.08  # dividend rate
sig = 0.1   # volatility
T = 1.0     # time to maturity
K = 100.0   # strike price

n_steps = 2000     # time steps
dt = T / n_steps    # time step size
#discount = np.exp(-mu * dt)    # discount factor per time step

# option payoff function

f_call = lambda x : np.where(x < K, 0.0, x - K)
f_put = lambda x : np.where(x < K, K - x, 0.0)

fpay0 = f_call

# underlying price range

a = 0.8 * K
b = 1.2 * K
x = np.linspace(a, b, 41)

# Finite difference algorithm (old)

g = fd_option0(fpay0, x, mu, div, sig, dt, n_steps, am=True, theta=0)
fd_price0 = g[0]

# Finite difference algorithm (new)

fpay = lambda x : fpay0(x[0])
fmu = lambda x : x[0][:, np.newaxis] * (mu - div)
fsig = lambda x : x[0][:, np.newaxis, np.newaxis] * sig
fdisc = lambda x : np.full(len(x[0]), mu)

g = fd_option(fpay, fmu, fsig, fdisc, x, dt, n_steps, am=True)
fd_price = g[0]

# Show results

fig, ax = plt.subplots()

ax.plot(x, fd_price, color='r', label='New')
ax.plot(x, fd_price0, color='g', linestyle='dashed', label='Old')

plt.xlabel("Underlying Price")
plt.ylabel("Option Price")

title = "American " + ("Call" if fpay == f_call else "Put") + " Option Values by Finite Difference"
plt.title(title)

leg = ax.legend()
offset = OffsetFrom(leg, (0.05, 0.0))
annotation = ("r = {0:5.2f}\n" + \
             "q = {1:5.2f}\n" + \
             "$\sigma$ = {2:5.2}\n" + \
             "K = {3:5.1f}\n" + \
             "T = {4:5.2f}").format(mu, div, sig, K, T)
ax.annotate(annotation, xy=(0,0),
            xycoords='figure fraction', xytext=(0,-20), textcoords=offset, 
            horizontalalignment='left', verticalalignment='top')


plt.show()

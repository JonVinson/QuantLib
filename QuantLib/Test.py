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
from FDPricing import *
from MCPricing import *
from MCPaths import *
from Regressors import *

mu = 0.05   # interest rate
div = 0.08  # dividend rate
sig = 0.1   # volatility
T = 1.0     # time to maturity
K = 100.0   # strike price

# option payoff function

f_call = lambda x : np.where(x < K, 0.0, x - K)
f_put = lambda x : np.where(x < K, K - x, 0.0)

f = f_call

# underlying price range

a = 0.8 * K
b = 1.2 * K
S = np.linspace(a, b, 21)

# transformation function for parametric regression

nodes = np.linspace(0, 2 * K, 41)
phi = lambda x : piecewise(x, nodes)

# Construct underlying paths

n_paths1 = 1000     # for first stage of Longstaff-Schwartz
n_paths2 = 10000    # for Tsistiklis-Van Roy and second stage of Longstaff-Schwartz

n_steps = 10        # time steps
dt = T / n_steps    # time step size
discount = np.exp(-mu * dt)    # discount factor per time step

paths1 = LogNormalPaths(n_paths1, n_steps, mu=mu-div, sig=sig, dt=dt)
paths2 = paths1.morepaths(n_paths2)

# Tsitsiklis-Van Roy algorithm

tvr_price = [TVR_param(s * paths2, f, phi, discount) for s in S]

# Longstaff-Schwartz algorithm

ls_price = [LS_param(s * paths1, s * paths2, f, phi, discount) for s in S]

# Finite difference algorithm

S2 = np.linspace(a, b, 41)

g = fd_option(f, S2, mu, div, sig, dt, n_steps, am=True)
fd_price = g[0]

# Black-Scholes price for European option

bs_price = bs_eur_call(K, S, mu, div, sig, T) if f == f_call else bs_eur_put(K, S, mu, div, sig, T)

# Show results

fig, ax = plt.subplots()

ax.plot(S, tvr_price, color='g', label='Tsitsiklis-Van Roy')
ax.plot(S, ls_price, color='b', label='Longstaff-Schwartz')
ax.plot(S2, fd_price, color='y', label='Finite Difference')
ax.plot(S, bs_price, color='r', label='Black-Scholes (Eur Opt)', linestyle='dashed')
plt.xlabel("Underlying Price")
plt.ylabel("Option Price")

title = "American " + ("Call" if f == f_call else "Put") + " Option Values by Various Methods"
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

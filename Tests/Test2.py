##########################################################################################################################
# This program demonstrates the use of the Tsitsiklis-Van Roy Algorithm, the Longstaff-Schwartz algorithm, the Andersen-
# Broadie algorithm, and a finite difference algorithm to estimate the value of an American option for a single underlying
# price. The Black-Scholes value of the corresponding European option is also calculated for comparison.
#
# The program depends on the following modules:
#
#   BlackSholes.py  : Black-Scholes prices
#   FDPRicing.py    : Finite difference algorithm
#   MCPricing.py    : TVR, LS, and AB algorithms
#   MCPaths.py      : Classes to contain Monte-Carlo paths and associated parameters
#   Processes.py    : Generate Monte-Carlo paths
#   Regressors.py   : Transformation functions used for parametric regression estimate of conditional expectation values
#
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

S = 105 # underlying price

# option payoff function

f_call = lambda x : np.where(x < K, 0.0, x - K)
f_put = lambda x : np.where(x < K, K - x, 0.0)

f = f_call

# transformation function for parametric regression

nodes = np.linspace(0, 2 * K, 41)
phi = lambda x : piecewise(x, nodes)

# Construct underlying paths

n_paths1 = 1000     # for first stage of Longstaff-Schwartz
n_paths2 = 10000     # for Tsitsiklis-Van Roy and second stage of Longstaff-Schwarz

n_steps = 10        # time steps

dt = T / n_steps    # time step size
discount = np.exp(-mu * dt)    # discount factor per time step

paths1 = LogNormalPaths(n_paths1, n_steps, mu=mu-div, sig=sig, dt=dt, X0=S)
#paths1 = SABRPaths(n_paths1, n_steps, mu=mu-div, alpha=0.5, beta=0.9, rho=-0.75, f0=S, sig0=sig, dt=dt)
#paths1 = HestonPaths(n_paths1, n_steps, mu-div, theta=sig**2, kappa=0.1, xi=0.5, rho=-0.75, S0=S, theta0=sig**2, dt=dt)

paths2 = paths1.morepaths(n_paths2)

# Tsitsiklis-Van Roy algorithm

tvr_price = TVR_param(paths2, f, phi, discount)

# Longstaff-Schwartz, Andersen-Broadie algorithms

(ab_price, ls_price, tau, rlist) = AB_param(paths1, paths2, f, phi, discount, full=True)

# Finite difference algorithm

a = 0.8 * K
b = 1.2 * K
S2 = np.linspace(a, b, 401)

g = fd_option(f, S2, mu, div, sig, 0.1 * dt, 10 * n_steps, True, 0.5)
fd_price = g[0, S2 == S][0]

# Black-Scholes price for European option

bs_price = bs_eur_put(K, S, mu, div, sig, T) if f == f_put else bs_eur_call(K, S, mu, div, sig, T)

# Print results

print("Parameters:")
print()

print("Int rate:\t\t{0:5.2f}".format(mu))
print("Div rate:\t\t{0:5.2f}".format(div))
print("Volatility:\t\t{0:5.2}".format(sig))
print("Strike:\t\t\t{0:5.1f}".format(K))
print("Maturity:\t\t{0:5.2f}".format(T))
print("Underlying price:\t{0:5.1f}".format(S))
print()

if (type(paths1) == SABRPaths):
    print("SABR Parameters:")
    print()
    print("Alpha:\t{0:5.2f}".format(paths1.alpha))
    print("Beta:\t{0:5.2f}".format(paths1.beta))
    print("Rho:\t{0:5.2f}".format(paths1.rho))
    print("Initial volatility:\t{0:5.2f}".format(paths1.Y[0,0]))
    print()

if (type(paths1) == HestonPaths):
    print("Heston Model Parameters:")
    print()
    print("Long-term variance:\t{0:5.2f}".format(paths1.theta))
    print("Mean return rate:\t{0:5.2f}".format(paths1.kappa))
    print("Volatility of variance:\t{0:5.2f}".format(paths1.xi))
    print("Correlation:\t\t{0:5.2f}".format(paths1.rho))
    print("Initial variance:\t{0:5.2f}".format(paths1.Y[0,0]**2))
    print()
    
print("Option Value Results:")
print()

print("Longstaff-Schwartz:\t{0:7.3f}".format(ls_price))
print("Andersen-Broadie:\t{0:7.3f}".format(ab_price))
print("Tsitsiklis-Van Roy:\t{0:7.3f}".format(tvr_price))
print("Finite difference:\t{0:7.3f}".format(fd_price))
print("Black-Scholes (Eur Opt):{0:7.3f}".format(bs_price))

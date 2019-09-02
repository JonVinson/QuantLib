#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

# FDPricing.py

import numpy as np
import scipy.sparse as sp
import warnings

from FiniteDifference import diff1, diff2

def fd_option(fpay, fmu, fsig, fdisc, x, dt, n, am = False, full = False):

    # solve backward equation for option with time-independent payoff function by finite difference scheme

    # Parameters:

    # fpay : payoff function
    # fmu, fsig, fdisc: drift rate, volatility, and discount functions
    # x : underlying value range
    # dt : t step size
    # n : time steps
    # am : American option flag

    # Return values:

    # g : Solution of equation
    # h : Excercise region of American option (1 = exercise, 0 = continuation)

    if not isinstance(x, list) and not isinstance(x, tuple):
        x = [x]

    ndim = len(x)

    nx = [np.size(xx) for xx in x]
    dx = [(np.max(xx) - np.min(xx)) / (np.size(xx) - 1) for xx in x]

    # Set terminal values

    F = np.array(fpay(x))

    mu = np.array(np.moveaxis(fmu(x), -1, 0))

    sig = np.array(fsig(x))
    ss = np.matmul(sig, np.moveaxis(sig, -2, -1))
    ss = np.array(np.moveaxis(ss, [-2, -1], [0, 1]))

    disc = np.array(fdisc(x))


    nx.insert(0, n + 1)

    g = np.zeros(tuple(nx))
    g[n] = np.array(F)
    
    h = np.zeros(tuple(nx))
    h[n] = 1.0
    
    # Begin backward loop

    for i in range(n - 1, -1, -1):

        g0 = g[i + 1]
        g[i] = g0 * (1.0 - disc * dt)

        for j in range(ndim):
            D = diff1(g0, j)
            g[i] = g[i] + (mu[j] * D / dx[j] + 0.5 * ss[j, j] * diff2(g0, j) / dx[j] ** 2) * dt
            for k in range(j):
                g[i] = g[i] + (ss[j, k] * diff1(D, k) / (dx[j] * dx[k])) * dt

        # American case:

        if (am):
            h[i, g[i] <= F] = 1.0
            g[i] = np.maximum(F, g[i])

    return (g, h) if full else g

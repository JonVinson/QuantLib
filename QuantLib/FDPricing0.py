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

def fd_option(f, x, mu, q, sig, dt, n, am = False, theta = 1.0, full = False):

    # solve backward equation for option with time-independent payoff function by finite difference scheme

    # Parameters:

    # f : payoff function
    # x : underlying value range
    # mu, q, sig: interest rate, dividend rate, volatility
    # dt : t step size
    # n : time steps
    # am : American option flag
    # theta : scheme parameter (1.0 = implicit, 0.0 = explict)

    # Return values:

    # g : Solution of equation
    # h : Excercise region of American option (1 = exercise, 0 = continuation)

    warnings.simplefilter("ignore", sp.SparseEfficiencyWarning)
    
    nx = np.size(x)
    dx = (np.max(x) - np.min(x)) / (nx - 1)

    # Set up matrix A, which transforms g[i] to estimate of (mu - q) * (dg/dx) + (1/2) * sig^2 * (d^2g/dx^2) - mu * g

    # Matrix elements for estimates in interior

    d1 = 0.5 * ((sig * x[1:] / dx) ** 2 - (mu - q) * x[1:] / dx)
    d2 = -(sig * x / dx) ** 2 - mu
    d3 = 0.5 * ((sig * x[:-1] / dx) ** 2 + (mu - q) * x[:-1] / dx)
    
    # Matrix elements for estimates at boundary

    A = sp.diags([d1, d2, d3], [-1, 0, 1], format='csr')

    A[0, :4] = (mu - q) * x[0] * np.array([-3., 4., -1., 0.]) / (2.0 * dx) + 0.5 * (sig * x[0]) ** 2 * np.array([2., -5., 4., -1.]) / (dx ** 2) - mu * np.array([1., 0., 0., 0.])
    A[-1, -4:] = (mu - q) * x[-1] * np.array([0, 1, -4, 3]) / (2.0 * dx) + 0.5 * (sig * x[-1]) ** 2 * np.array([-1, 4, -5, 2]) / (dx ** 2) - mu * np.array([0, 0, 0, 1])

    # from A and theta, calculate matrix B, which transforms g[i] to estimate of g[i-1]

    A0 = sp.eye(nx) + (1.0 - theta) * dt * A 
    A1 = sp.eye(nx) - theta * dt * A

    B = sp.linalg.inv(A1) * A0

    # Set terminal values

    F = np.array(f(x))
    
    g = np.zeros((n + 1, nx))
    g[n] = np.array(F)
    
    h = np.zeros((n + 1, nx))
    h[n, :] = 1.0
    
    # Begin backward loop

    for i in range(n - 1, -1, -1):

        g[i] = B.dot(g[i + 1])

        # American case:

        if (am):
            h[i, g[i] <= F] = 1.0
            g[i] = np.maximum(F, g[i])

    return (g, h) if full else g

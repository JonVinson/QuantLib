#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

# MCPaths.py

import numpy as np
from Processes import *

class MCPaths:

    # Base class for MC path containers

    def subpaths(self, path, step, n_subpaths, n_substeps):
        return NotImplemented

    def morepaths(self, n_paths):
        return self.subpaths(0, 0, n_paths, self.n_steps)

    def multiple(self, a):
        return NotImplemented
    
    def __mul__(self, a):
        return self.multiple(a)

    def __rmul__(self, a):
        return self.multiple(a)

########################################################################################################
    
class NormalPaths(MCPaths):

    # Container for normal (Brownian) paths

    def __init__(self, n_paths = 0, n_steps = 0, X0 = 0.0, mu = 0.0, sig = 1.0, dt = 1.0):
        self.X = None
        self.mu = mu
        self.sig = sig
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        if n_paths > 0 and n_steps > 0:
            self.X = brownian(n_paths, n_steps, X0, mu, sig, dt)

    def subpaths(self, path, step, n_subpaths, n_substeps):
        return NormalPaths(n_subpaths, n_substeps, self.X[path, step], self.mu, self.sig, self.dt)

    def multiple(self, a):
        p = NormalPaths()
        p.X = a * self.X
        p.mu = a * self.mu
        p.sig = a * self.sig
        p.n_paths = self.n_paths
        p.n_steps = self.n_steps
        p.dt = self.dt
        return p

###################################################################################

class LogNormalPaths(MCPaths):

    # Container for log-normal (geometric Brownian) paths

    def __init__(self, n_paths = 0, n_steps = 0, X0 = 1.0, mu = 0.0, sig = 1.0, dt = 1.0):
        self.X = None
        self.mu = mu
        self.sig = sig
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        if n_paths > 0 and n_steps > 0:
            self.X = geometric_brownian(n_paths, n_steps, X0, mu, sig, dt)

    def subpaths(self, path, step, n_subpaths, n_substeps):
        return LogNormalPaths(n_subpaths, n_substeps, self.X[path, step], self.mu, self.sig, self.dt)

    def multiple(self, a):
        p = LogNormalPaths()
        p.X = a * self.X
        p.mu = self.mu
        p.sig = self.sig
        p.n_paths = self.n_paths
        p.n_steps = self.n_steps
        p.dt = self.dt
        return p

###################################################################################

class SABRPaths(MCPaths):

    # Container for value and volatility paths from SABR model

    def __init__(self, n_paths = 0, n_steps = 0, mu = 0.0, alpha = 0.0, beta = 1.0, rho = 0.0, f0 = 0.0, sig0 = 0.0, dt = 1.0):
        self.X = None
        self.Y = None
        self.Z = None
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        if n_paths > 0 and n_steps > 0:
            (self.X, self.Y, self.Z) = sabr(n_paths, n_steps, mu, alpha, beta, rho, f0, sig0, dt)

    def subpaths(self, path, step, n_subpaths, n_substeps):
        return SABRPaths(n_subpaths, n_substeps, self.mu, self.alpha, self.beta, self.rho, self.X[path, step], self.Y[path, step], self.dt)

    def recalculate(self, a):
        X1 = np.zeros(np.shape(self.X))
        X1[:, 0] = a * self.X[:, 0]
        n_steps = np.size(X1, 1) - 1
        for i in range(n_steps):
            X1[:, i + 1] = X1[:, i] + self.Y[:, i] * np.power(X1[:, i], self.beta) * self.Z[:, i]
        if self.mu != 0.0:
            d = np.exp(self.mu * self.dt * np.array(range(n_steps + 1)))
            X1 = X1 * d
        return X1

    def multiple(self, a):
        p = SABRPaths()
        p.X = self.recalculate(a)
        p.Y = self.Y
        p.mu = self.mu
        p.alpha = self.alpha
        p.beta = self.beta
        p.rho = self.rho
        p.n_paths = self.n_paths
        p.n_steps = self.n_steps
        p.dt = self.dt
        return p



###################################################################################

class HestonPaths(MCPaths):

    # Container for value and volatility paths from Heston model
    
    def __init__(self, n_paths = 0, n_steps = 0, mu = 0.0, theta = 1.0, kappa = 1.0, xi = 1.0, rho = 0.0, S0 = 1.0, theta0 = 0.0, dt = 1.0):
        self.X = None
        self.Y = None
        self.mu = mu
        self.theta = theta
        self.kappa = kappa
        self.xi = xi
        self.rho = rho
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        if n_paths > 0 and n_steps > 0:
            (self.X, self.Y) = heston(n_paths, n_steps, mu, theta, kappa, xi, rho, S0, theta0, dt)

    def subpaths(self, path, step, n_subpaths, n_substeps):
        return HestonPaths(n_subpaths, n_substeps, self.mu, self.theta, self.kappa, self.xi, self.rho, self.X[path, step], self.Y[path, step] ** 2, self.dt)

    def multiple(self, a):
        p = HestonPaths()
        p.X = a * self.X
        p.Y = self.Y
        p.mu = self.mu
        p.theta = self.theta
        p.kappa = self.kappa
        p.xi = self.xi
        p.rho = self.rho
        p.n_paths = self.n_paths
        p.n_steps = self.n_steps
        p.dt = self.dt
        return p

########################################################################################################


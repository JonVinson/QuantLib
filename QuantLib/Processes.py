#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

import numpy as np
    
def gaussian(n_paths, n_steps, mu = 0.0, sig = 1.0):

    # normally distributed random numbers

    # n_paths : number of M.C. paths
    # n_steps : number of time steps
    # mu : mean
    # sig : standard deviation
    
    return np.random.normal(mu, sig, size=(n_paths, n_steps))

########################################################################################################
    
def multivariate_gaussian(n_paths, n_steps, mu, cov):

    # normally distributed multivariate random numbers

    # n_paths : number of M.C. paths
    # n_steps : number of time steps
    # mu : mean
    # cov : covariance matrix
    
    return np.random.multivariate_normal(mu, cov, size=(n_paths, n_steps))

########################################################################################################
    
def brownian(n_paths, n_steps, X0 = 0.0, mu = 0.0, sig = 1.0, dt = 1.0):

    # brownian paths

    # n_paths : number of M.C. paths
    # n_steps : number of time steps
    # X0 : initial path value
    # mu, sig : drift and stardard deviation
    # dt: time step size
    
    # gaussian samples, scaled by step size

    z = gaussian(n_paths, n_steps, mu * dt, sig * np.sqrt(dt))

    # return cumulated gaussians

    return X0 + np.hstack((np.zeros((n_paths, 1)), np.cumsum(z, axis=1)))

##########################################################################################################

def geometric_brownian(n_paths, n_steps, X0 = 1.0, mu = 0.0, sig = 1.0, dt = 1.0):

    # geometric brownian paths

    # n_paths : number of M.C. paths
    # n_steps : number of time steps
    # X0 : initial path value
    # mu, sig : drift and stardard deviation
    # dt: time step size

    # adjust mu for log of path values

    mu_log = mu - 0.5 * sig ** 2

    # gaussian samples, scaled by step size

    z = gaussian(n_paths, n_steps, mu_log * dt, sig * np.sqrt(dt))

    # return exponential of cumulated gaussians 

    return X0 * np.hstack((np.ones((n_paths, 1)), np.exp(np.cumsum(z, axis=1))))

##########################################################################################################

def heston(n_paths, n_steps, mu = 0.0, theta = 0.0, kappa = 0.0, xi = 0.0, rho = 0.0, S0 = 1.0, theta0 = 0.0, dt = 1.0):
    
    # Heston stochastic volatility model

    # Simulates processes
    #
    #   dS(t) = mu * S(t) * dt + xi * n(t)^(1/2) * S(t) * dW
    #
    #   dn(t) = kappa * (theta - n(t)) * dt  + xi * n(t)^(1/2) * dX
    #
    # where W and X are Wiener process with correlation rho

    # Other parameters:

    # n_paths, n_steps : Number of paths and path steps
    # dt: time step size
    
    # Return values: value processes S(t) and volatility processes s(t) = n(t)^(1/2)

    h = np.zeros((n_paths, n_steps + 1))
    h[:, 0] = theta0

    scale = np.sqrt(dt)
    
    z1 = np.random.normal(size=(n_paths, n_steps), scale=scale)
    z2 = np.random.normal(size=(n_paths, n_steps), scale=scale)
 
    z = rho * z1 + np.sqrt(1.0 - rho ** 2) * z2
    
    for i in range(n_steps):
        h[:, i + 1] =  h[:, i] + kappa * (theta - h[:, i]) * dt + xi * np.sqrt(h[:, i]) * z1[:, i]
        h[:, i + 1] = np.maximum(h[:, i + 1], 0.0)

    sig = np.sqrt(h)

    z = (mu - 0.5 * h[:,:-1]) * dt + sig[:,:-1] * z;

    S = S0 * np.exp(np.hstack((np.zeros((n_paths, 1)), np.cumsum(z, axis=1))))

    return (S, sig)

##########################################################################################################

def sabr(n_paths, n_steps, mu = 0.0, alpha = 0.0, beta = 1.0, rho = 0.0, f0 = 0.0, sig0 = 0.0, dt = 1.0, Z1 = None, Z2 = None):

    # SABR stochastic volatility model

    # Simulates processes
    #
    #   dF(t) = s(t) * F(t)^beta * dW(t)
    #
    #   ds(t) = alpha * s(t) * dX(t)
    #
    # where W and X are Wiener process with correlation rho

    # Other parameters:
    
    # Z1, Z2: arrays of normally distributed random numbers
    # n_paths, n_steps : Number of paths and path steps (if Z1 and Z2 not supplied)
    # dt: time step size
    
    # Return values: forward value processes F(t), volatility processes s(t), and gaussian process for values
    # (in case value process needs to be recalculated for a different initial value)
    
    scale = np.sqrt(dt)

    if not (Z1 is None or Z2 is None):
        n_paths = np.size(Z1, 0)
        n_steps = np.size(Z1, 1)
        z1 = scale * Z1
        z2 = scale * (rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2)
    elif n_paths > 0 and n_steps > 0:
        paths = multivariate_gaussian(n_paths, n_steps, [0.0, 0.0], [[1.0, rho], [rho, 1.0]])
        paths[np.abs(paths) > 6.0] = 0.0
        z1 = paths[:, :, 0] * scale
        z2 = paths[:, :, 1] * scale
    else:
        return None

    z = alpha * z1 - 0.5 * alpha ** 2 * dt
    sig = sig0 * np.exp(np.hstack((np.zeros((n_paths, 1)), np.cumsum(z, axis=1))))

    f = np.zeros((n_paths, n_steps + 1))

    f[:, 0] = f0

    for i in range(n_steps):
        f[f[:, i] < 0.0, i] = 0.0
        f[:, i + 1] = f[:, i] + sig[:, i] * (np.power(f[:, i], beta) * z2[:, i] + 0.5 * beta * np.power(f[:, i], 2 * beta - 1) * (z2[:, i] ** 2 - dt))

    if mu != 0.0:
        d = np.exp(mu * dt * np.array(range(n_steps + 1)))
        f = f * d

    return (f, sig, z2)

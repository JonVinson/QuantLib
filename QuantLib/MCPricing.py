#######################################################################################################################
# This file is the property of Jon discount. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon discount. Vinson                                                                                     #
#######################################################################################################################

import numpy as np
from sklearn import linear_model as lm

def TVR_param(paths, f, phi, discount = 1.0):

    # Tsitsiklis-Van Roy algorithm with conditional expectation values estimated by parametric regression

    # Parameters:

    # paths :       underlying value process
    # f :           payoff function
    # phi :         regressor function
    # discount :    step discount factor

    # Return value: option value

    n_steps = paths.n_steps

    F = f(paths.X)
    regressors = phi(paths.X)

    r = lm.LinearRegression()
    
    V = F[:, n_steps]

    for i in range(n_steps - 1, 0, -1):
        X = regressors[:, :, i]
        r.fit(X, V)
        V = r.predict(X)
        V = np.maximum(F[:, i], discount * V)
    
    return np.maximum(discount * np.mean(V), F[0, 0])
    
######################################################################################################

def LS_param(paths1, paths2, f, phi, discount = 1.0, full = False):

    # Longstaff-Schwartz algorithm with conditional expectation values estimated by parametric regression

    # Parameters:

    # paths1 :      underlying value process for stopping time estimation
    # paths2 :      underlying value process for final valuation
    # f :           payoff function
    # phi :         regressor function
    # discount :    step discount factor
    # full:         flag for optional return values

    # Return values:

    # val :     option value
    # tau :     stopping time indexes of path sample 2 (optional)
    # rlist :   Scikit-learn linear regression models for later use (optional)

    # Step 1: Use TVR algorithm with small MC sample to estimate exercise region, captured by regression coefficients

    n_steps = paths1.n_steps

    F = f(paths1.X)
    regressors = phi(paths1.X)

    rlist = []

    V = F[:, n_steps]

    for i in range(n_steps - 1, 0, -1):
        X = regressors[:, :, i]
        r = lm.LinearRegression()
        rlist.append(r)    
        r.fit(X, V)
        V = r.predict(X)
        V = np.maximum(F[:, i], discount * V)
    
    rlist.reverse()

    # Step 2: Use larger MC saample and regression coefficents to estimate discounted expected value at stopping time.
    # Result is low-biased estimate of option value

    (val, tau) = LS_step2(paths2, f, phi, rlist, discount, full=True)

    return (val, tau, rlist) if full else val

##########################################################################################################

def LS_step2(paths, f, phi, rlist, discount, continue_zero=False, full=False):

    # Use supplied paths and regression coefficents to estimate expected value at stopping time.
    # Used in second step of Longstaff-Schartz algorithm, and to estimate conditional expectations from
    # subpaths in Andersen-Broadie algorithm

    # Parameters:

    # paths :          underlying value process for stopping time estimation
    # f :               payoff function
    # phi :             regressor function
    # rlist :           regression coefficients from first LS step
    # discount :        step discount factor
    # continue_zero :   assume stopping time greater than zero (used for "plus one" expectation values in AB alrorithm)
    # full:             flag for optional return values

    # Return values:

    # val :     Option value
    # tau :     stopping time indexes of path sample (optional)

    n_paths = paths.n_paths
    n_steps = paths.n_steps

    F = f(paths.X)
    regressors = phi(paths.X)

    tau = np.zeros((n_paths, n_steps + 1), dtype=np.int32)
    tau[:, n_steps] = n_steps
    
    for i in range(n_steps - 1, 0, -1):
        X = regressors[:, :, i]
        r = rlist[i - 1]
        V = r.predict(X)
        tau[:, i] = tau[:, i + 1]
        j_stop = discount * V <= F[:, i]
        tau[j_stop, i] = i

    val = np.mean(np.array([F[j, tau[j, 1]] * discount ** tau[j, 1] for j in range(n_paths)]))

    tau[:, 0] = tau[:, 1]

    if val <= F[0, 0] and not continue_zero:
        val = F[0, 0]
        tau[:, 0] = 0

    return (val, tau) if full else val

###########################################################################################################

def AB_param(paths1, paths2, f, phi, discount = 1.0, np3 = [300, 300, 300], full = False):

    # Andersen-Broadie algorithm described by Andersen and Broadie, "Primal-Dual Simulation Algorithm for
    # Pricing Multidimensional American Options", Management Science, Vol. 50, No. 9, September 2009
    # This implementation is based on J. Guyon & P. Henry-Labordere, "Nonlinear Option Pricing".
    
    # Parameters:

    # paths1 :      underlying value process for LS stopping time estimation
    # paths2 :      underlying value process for LS valuation
    # f :           payoff function
    # phi :         regressor function
    # discount :    step discount factor
    # np3:          number of subpaths for estimation of expectation values in AB calculation
    # full:         flag for optional return values

    # Return values:

    # ab_val :  AB option value
    # ls_val :  LS option value (optional)
    # tau :     stopping time indexes of path sample 3 (optional)
    # rlist :   Scikit-learn linear regression models for later use (optional)

    # Step 1: Use Longstaff-Schwartz algorithm to estimate excercise region and LS optional value

    (ls_val, tau, rlist) = LS_param(paths1, paths2, f, phi, discount, True)

    # Step 2: Generate independent sample to estimate optimal martingale

    paths3 = paths1.morepaths(np3[0])

    n_paths = paths3.n_paths
    n_steps = paths3.n_steps

    F = f(paths3.X)

    DV = np.empty(n_steps + 1)
    EDV = np.empty(n_steps + 1)
    M = np.zeros(n_steps + 1)
    V = np.empty(n_paths)

    discounts = discount ** range(0, n_steps + 1)

    # Get stopping time indexes for new sample

    (v, tau) = LS_step2(paths3, f, phi, rlist, discount, full=True)

    # Begin martingale estimation

    for i_path in range(n_paths):

        DV[:] = 0.0
        EDV[:] = 0.0

        # Indvidual terms (6.35), (6.36), and (6.37) in Guyon & Henry-Labordere

        for j_step in range(n_steps + 1):

            if tau[i_path, j_step] > j_step: # continuation case
                
                subpaths = paths3.subpaths(i_path, j_step, np3[1], n_steps - j_step)
                DV[j_step] = LS_step2(subpaths, f, phi, rlist[j_step:], discount) * discounts[j_step]

            else: # exercise case
                
                DV[j_step] = F[i_path, j_step] * discounts[j_step]
                if j_step < n_steps:
                    subpaths = paths3.subpaths(i_path, j_step, np3[2], n_steps - j_step)
                    EDV[j_step] = LS_step2(subpaths, f, phi, rlist[j_step:], discount, continue_zero=True) * discounts[j_step]

        # Martingale increments (6.34)

        for j_step in range(n_steps):
            M[j_step + 1] = DV[j_step + 1] - (DV[j_step] if tau[i_path, j_step] > j_step else EDV[j_step])

        # path value (6.33)

        V[i_path] = np.max(discount * F[i_path, :] - np.cumsum(M))

    # final option value (6.33)

    ab_val = np.mean(V)

    return (ab_val, ls_val, tau, rlist) if full else ab_val

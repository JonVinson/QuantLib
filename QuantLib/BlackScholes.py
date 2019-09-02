#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

# BlackScholes.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Price functions

def bs_eur_call(K, S, r, q, sig, T):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return norm.cdf(d1) * S * np.exp(-q * T) - norm.cdf(d2) * K * np.exp(-r * T)

#####################################################################################3

def bs_eur_put(K, S, r, q, sig, T):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return -norm.cdf(-d1) * S * np.exp(-q * T) + norm.cdf(-d2) * K * np.exp(-r * T)

#####################################################################################3

# Implied volatility functions

def bs_eur_call_ivol(P, K, S, r, q, T, sig0 = 1.0):
    f = lambda sig : bs_eur_call(K, S, r, q, sig, T) - P
    try:
        return brentq(f, 0.01 * sig0, 10.0 * sig0)
    except:
        return np.nan

#####################################################################################3

def bs_eur_put_ivol(P, K, S, r, q, T, sig0 = 1.0):
    f = lambda sig : bs_eur_put(K, S, r, q, sig, T) - P
    try:
        return brentq(f, 0.01 * sig0, 4.0 * sig0)
    except:
        return np.nan


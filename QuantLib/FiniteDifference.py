#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

import numpy as np

def d1(a):
    # first derivative of array a along 0th axis
    b = np.empty(np.shape(a))
    b[0] = -1.5 * a[0] + 2.0 * a[1] - 0.5 * a[2]
    b[-1] = 0.5 * a[-3] - 2.0 * a[-2] + 1.5 * a[-1]
    b[1:-1] = 0.5 * (a[2:] - a[:-2])
    return b

def d2(a):
    # second derivative of array a along 0th axis
    b = np.empty(np.shape(a))
    b[0] = 2.0 * a[0] - 5.0 * a[1] + 4.0 * a[2] - a[3]
    b[-1] = 2.0 * a[-1] - 5.0 * a[-2] + 4.0 * a[-3] - a[-4]
    b[1:-1] = a[2:] - 2.0 * a[1:-1] + a[:-2]
    return b

def diff1(a, n):
    # first derivative of array a along nth axis
    b = np.moveaxis(a, n, 0)
    return np.moveaxis(d1(b), 0, n)

def diff2(a, n):
    # second derivative of array a along nth axis
    b = np.moveaxis(a, n, 0)
    return np.moveaxis(d2(b), 0, n)


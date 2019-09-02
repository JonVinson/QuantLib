#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

import numpy as np

def piecewise(x, nodes):

    # Used for piecewise linear regression

    n_nodes = len(nodes)
    diag = np.identity(n_nodes)
    res = np.empty((np.size(x, 0), n_nodes, np.size(x, 1)))
    for i in range(n_nodes):
        res[:, i, :] = np.interp(x, nodes, diag[i])
    return res

def gaussian(x, nodes):
    sig = (nodes[1] - nodes[0]) / np.sqrt(6.0)
    return np.exp(-0.5 * ((x[:, np.newaxis, :] - nodes[:, np.newaxis]) / sig) ** 2)

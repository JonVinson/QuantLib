#######################################################################################################################
# This file is the property of Jon D. Vinson, and is provided for evaluation and testing purposes only.               #
# You may not transmit or provide copies of this file to anyone outside your organization without express permission. #
# This file is provided AS IS, and use of this file is AT YOUR OWN RISK.                                              #
#                                                                                                                     #
# Copyright 2018 by Jon D. Vinson                                                                                     #
#######################################################################################################################

# FDSolve.py

import numpy as np
import scipy.linalg as lin

class FDSolver2D:

    _stencil_a = np.array([0, 0.5])
    _stencil_b = -np.flip(_stencil_a, 0)
    _stencil_c = np.array([-0.5, 0, 0.5])

    _stencil_2a = np.array([-2, 1])
    _stencil_2b = np.flip(_stencil_2a, 0)
    _stencil_2c = np.array([1, -2, 1])

    def __init__(self, shape, intervals, condition = None, mu = None, ss = None):

        [self._nx, self._ny, self._nt] = shape
        [self._dx, self._dy, self._dt] = intervals

        self._cond = np.zeros((self._nx, self._ny)) if condition is None else condition
        self._mu = np.zeros((self._nx, self._ny)) if mu is None else mu
        self._ss = np.zeros((self._nx, self._ny)) if ss is None else ss

        self._a = np.zeros((self._nx, self._ny, self._nx, self._ny))
        self._g = np.empty((self._nt, self._nx * self._ny))

        self._nt -= 1

#--------------------------------------------------------------------------------#

    def SetCoefficients(self, coeffiecents):
        
        [mu, ss] = coeffiecents

        if self._mu is None or np.shape(self._mu) != np.shape(mu):
            self._mu = mu
        else:
            self._mu[:] = mu

        if self._ss is None or np.shape(self._ss) != np.shape(ss):
            self._ss = ss
        else:
            self._ss[:] = ss

#--------------------------------------------------------------------------------#
    def SetCondition(self, cond):
        if self._cond is None or np.shape(self._cond) != np.shape(cond):
            self._cond = cond
        else:
            self._cond[:] = cond

    def Condition(self):
        return self._cond;
#--------------------------------------------------------------------------------#
    def _validateSetup(self):
        valid = True
        if self._cond is None:
            print("condition not set")
            valid = False
        if self._ss is None and self._mu is None:
            print("ss and/or mu not set")
            valid = False
        if self._ss is not None and self._mu is not None and np.shape(self._ss) != np.shape(self._mu):
            print("mu and ss are different shapes")
            valid = False
        if self._ss is not None and np.shape(self._ss) != np.shape(self._cond):
            print("condition and ss are different shapes")
            valid = False
        if self._mu is not None and np.shape(self._mu) != np.shape(self._cond):
            print("condition and mu are different shapes")
            valid = False
        return valid

#--------------------------------------------------------------------------------#
    def SolveBackward(self):

        if not self._validateSetup():
            return None

        self._a[:] = 0

        nx = self._nx
        dx = self._dx
        ny = self._ny
        dy = self._dy
        nt = self._nt
        dt = self._dt
        term = self._cond
        mu = self._mu
        ss = self._ss
        a = self._a
        g = self._g

        stencil_a = self._stencil_a
        stencil_b = self._stencil_b
        stencil_c = self._stencil_c
        stencil_2a = self._stencil_2a
        stencil_2b = self._stencil_2b
        stencil_2c = self._stencil_2c

        if not mu is None:

            for j in range(ny):
                a[0, j, :2, j] += mu[:2, j, 0] * stencil_a / dx
                a[-1, j, -2:, j] += mu[-2:, j, 0] * stencil_b / dx
                for i in range(1, nx - 1):
                    a[i, j, i-1:i+2, j] += mu[i-1:i+2, j, 0] * stencil_c / dx

            for i in range(nx):
                a[i, 0, i, :2] += mu[i, :2, 1] * stencil_a / dy
                a[i, -1, i, -2:] += mu[i, -2:, 1] * stencil_b / dy
                for j in range(1, ny - 1):
                    a[i, j, i, j-1:j+2] += mu[i, i-1:i+2, 1] * stencil_c / dy

        if not ss is None:

            dx2 = dx * dx

            for j in range(ny):
                a[0, j, :4, j] += 0.5 * ss[0, j, 0, 0] * stencil_2a / dx2
                a[-1, j, -4:, j] += 0.5 * ss[-1, j, 0, 0] * stencil_2b / dx2
                for i in range(1, nx - 1):
                    a[i, j, i-1:i+2, j] += 0.5 * ss[i, j, 0, 0] * stencil_2c / dx2

            dy2 = dy * dy

            for i in range(nx):
                a[i, 0, i, :4] += 0.5 * ss[i, 0, 1, 1] * stencil_2a / dy2
                a[i, -1, i, -4:] += 0.5 * ss[i, -1, 1, 1] * stencil_2b / dy2
                for j in range(1, ny - 1):
                    a[i, j, i, j-1:j+2] += 0.5 * ss[i, j, 1, 1] * stencil_2c / dy2

            dxy = dx * dy

            stencil = stencil_a[:, np.newaxis] * stencil_a
            a[0, 0, :3, :3] += ss[0, 0, 1, 0] * stencil / dxy

            stencil = stencil_a[:, np.newaxis] * stencil_b
            a[0, -1, :3, -3:] += ss[0, -1, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_a
            a[-1, 0, -3:, :3] += ss[-1, 0, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_b
            a[-1, -1, -3:, -3:] += ss[-1, -1, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_a
            for i in range(1, nx - 1):
                a[i, 0, i-1:i+2, :3] += ss[i, 0, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_b
            for i in range(1, nx - 1):
                a[i, -1, i-1:i+2, -3:] += ss[i, -1, 1, 0] * stencil / dxy

            stencil = stencil_a[:, np.newaxis] * stencil_c
            for j in range(1, ny - 1):
                a[0, j, :3, j-1:j+2] += ss[0, j, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_c
            for j in range(1, ny - 1):
                a[-1, j, -3:, j-1:j+2] += ss[-1, j, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_c
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    a[i, j, i-1:i+2, j-1:j+2] += ss[i, j, 1, 0] * stencil / dxy

        a[:] = -dt * a

        for i in range(nx):
            for j in range(ny):
                a[i, j, i, j] += 1

        A = np.reshape(a, (nx * ny, nx * ny))
        A[:] = lin.inv(A, overwrite_a=True)

        g[nt] = term.reshape(nx * ny)

        for i in range(nt, 0, -1):
            np.dot(A, g[i], g[i - 1])
            g[i - 1][g[i - 1] < 0.0] = 0
            g[i - 1] /= np.sum(g[i - 1])

#------------------------------------------------------------------------------------------------------------------------#

    def SolveForward(self):

        if not self._validateSetup():
            return None

        self._a[:] = 0

        nx = self._nx
        dx = self._dx
        ny = self._ny
        dy = self._dy
        nt = self._nt
        dt = self._dt
        init = self._cond
        mu = self._mu
        ss = self._ss
        a = self._a
        g = self._g

        stencil_a = self._stencil_a
        stencil_b = self._stencil_b
        stencil_c = self._stencil_c
        stencil_2a = self._stencil_2a
        stencil_2b = self._stencil_2b
        stencil_2c = self._stencil_2c

        if not mu is None:

            for j in range(ny):
                a[0, j, :2, j] -= mu[:2, j, 0] * stencil_a / dx
                a[-1, j, -2:, j] -= mu[-2:, j, 0] * stencil_b / dx
                for i in range(1, nx - 1):
                    a[i, j, i-1:i+2, j] -= mu[i-1:i+2, j, 0] * stencil_c / dx

            for i in range(nx):
                a[i, 0, i, :2] -= mu[i, :2, 1] * stencil_a / dy
                a[i, -1, i, -2:] -= mu[i, -2:, 1] * stencil_b / dy
                for j in range(1, ny - 1):
                    a[i, j, i, j-1:j+2] -= mu[i, j-1:j+2, 1] * stencil_c / dy

        if not ss is None:

            dx2 = dx * dx

            for j in range(ny):
                a[0, j, :2, j] += 0.5 * ss[:2, j, 0, 0] * stencil_2a / dx2
                a[-1, j, -2:, j] += 0.5 * ss[-2:, j, 0, 0] * stencil_2b / dx2
                for i in range(1, nx - 1):
                    a[i, j, i-1:i+2, j] += 0.5 * ss[i-1:i+2, j, 0, 0] * stencil_2c / dx2

            dy2 = dy * dy

            for i in range(nx):
                a[i, 0, i, :2] += 0.5 * ss[i, :2, 1, 1] * stencil_2a / dy2
                a[i, -1, i, -2:] += 0.5 * ss[i, -2:, 1, 1] * stencil_2b / dy2
                for j in range(1, ny - 1):
                    a[i, j, i, j-1:j+2] += 0.5 * ss[i, j-1:j+2, 1, 1] * stencil_2c / dy2

            dxy = dx * dy

            stencil = stencil_a[:, np.newaxis] * stencil_a
            a[0, 0, :2, :2] += 0.5 * ss[:2, :2, 1, 0] * stencil / dxy

            stencil = stencil_a[:, np.newaxis] * stencil_b
            a[0, -1, :2, -2:] += 0.5 * ss[:2, -2:, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_a
            a[-1, 0, -2:, :2] += 0.5 * ss[-2:, :2, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_b
            a[-1, -1, -2:, -2:] += 0.5 * ss[-2:, -2:, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_a
            for i in range(1, nx - 1):
                a[i, 0, i-1:i+2, :2] += 0.5 * ss[i-1:i+2, :2, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_b
            for i in range(1, nx - 1):
                a[i, -1, i-1:i+2, -2:] += 0.5 * ss[i-1:i+2, -2:, 1, 0] * stencil / dxy

            stencil = stencil_a[:, np.newaxis] * stencil_c
            for j in range(1, ny - 1):
                a[0, j, :2, j-1:j+2] += 0.5 * ss[:2, j-1:j+2, 1, 0] * stencil / dxy

            stencil = stencil_b[:, np.newaxis] * stencil_c
            for j in range(1, ny - 1):
                a[-1, j, -2:, j-1:j+2] += 0.5 * ss[-2:, j-1:j+2, 1, 0] * stencil / dxy

            stencil = stencil_c[:, np.newaxis] * stencil_c
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    a[i, j, i-1:i+2, j-1:j+2] += 0.5 * ss[i-1:i+2, j-1:j+2, 1, 0] * stencil / dxy

        a[:] = -dt * a

        for i in range(nx):
            for j in range(ny):
                a[i, j, i, j] += 1

        A = np.reshape(a, (nx * ny, nx * ny))
        A[:] = lin.inv(A, overwrite_a=True)

        g[0] = init.reshape(nx * ny)

        for i in range(nt):
            np.dot(A, g[i], g[i + 1])
            g[i + 1][g[i + 1] < 0.0] = 0
            g[i + 1] /= np.sum(g[i + 1])

#-------------------------------------------------------------------------------------------------------------------------#

    def Solution(self):
        return np.reshape(self._g, (self._nt + 1, self._nx, self._ny))


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

    stencil_a = np.array([0, 0.5])
    stencil_b = -np.flip(stencil_a, 0)
    stencil_c = np.array([-0.5, 0, 0.5])

    stencil_2a = np.array([-2, 1])
    stencil_2b = np.flip(stencil_2a, 0)
    stencil_2c = np.array([1, -2, 1])

    def __init__(self, nx, dx, ny, dy, nt, dt, cond = None, mu = None, ss = None):

        self.nx = nx
        self.dx = dx
        self.ny = ny
        self.dy = dy
        self.nt = nt
        self.dt = dt

        self.cond = cond
        self.mu = mu
        self.ss = ss

        self.a = np.zeros((nx, ny, nx, ny))
        self.g = np.empty((nt + 1, nx * ny))

#--------------------------------------------------------------------------------#
    
    def Set(self, cond, mu, ss):
        
        if self.cond is None or np.shape(self.cond) != np.shape(cond):
            self.cond = cond
        else:
            self.cond[:] = cond

        if self.mu is None or np.shape(self.mu) != np.shape(mu):
            self.mu = mu
        else:
            self.mu[:] = mu

        if self.ss is None or np.shape(self.ss) != np.shape(ss):
            self.ss = ss
        else:
            self.ss[:] = ss

        self.a[:] = 0

#--------------------------------------------------------------------------------#
    
    def SolveBackward(self):

        nx = self.nx
        dx = self.dx
        ny = self.ny
        dy = self.dy
        nt = self.nt
        dt = self.dt
        term = self.cond
        mu = self.mu
        ss = self.ss
        a = self.a
        g = self.g

        stencil_a = self.stencil_a
        stencil_b = self.stencil_b
        stencil_c = self.stencil_c
        stencil_2a = self.stencil_2a
        stencil_2b = self.stencil_2b
        stencil_2c = self.stencil_2c

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

        nx = self.nx
        dx = self.dx
        ny = self.ny
        dy = self.dy
        nt = self.nt
        dt = self.dt
        init = self.cond
        mu = self.mu
        ss = self.ss
        a = self.a
        g = self.g

        stencil_a = self.stencil_a
        stencil_b = self.stencil_b
        stencil_c = self.stencil_c
        stencil_2a = self.stencil_2a
        stencil_2b = self.stencil_2b
        stencil_2c = self.stencil_2c

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
        return np.reshape(self.g, (self.nt + 1, self.nx, self.ny))


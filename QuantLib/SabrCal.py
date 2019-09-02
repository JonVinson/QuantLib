import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares, minimize, brute

from Processes import *
from BlackScholes import *
from FDPricing import *
from FDSolve import *

################################################################################################################

def sabr_prices_mc(alpha, beta, rho, sig0, f0, mu, K, T, Z1, Z2, dt):

    (S, a, z) = sabr(0, 0, alpha=alpha, beta=beta, rho=rho, sig0=sig0, f0=f0, mu=mu, dt=dt, Z1=Z1, Z2=Z2)

    D = np.exp(-mu * T)

    P = np.zeros((len(K), len(T)))

    for i in range(len(K)):
        
        f = lambda x : np.where(x < K[i], 0.0, x - K[i])

        for j in range(len(T)):

            k = int(np.rint((T[j] / dt)))

            P[i, j] = np.mean(f(S[:, k])) * D[j]

    return P

##############################################################################################################

def sabr_prices_fd(alpha, beta, rho, sig0, f0, mu, K, T):

    a = 0.5 * f0
    b = 1.5 * f0
    F = np.linspace(a, b, 41)

    # underlying vol range
    a = 0.0
    b = 2.0 * sig0
    sig = np.linspace(a, b, 41)

    # Finite difference algorithm (new)

    def fmu(x):
        [F, s] = x
        return np.zeros((len(F), len(s), 2))

    def fsig(x):
        [F, s] = x
        sig = np.zeros((len(F), len(s), 2, 2))
        sig[:, :, 0, 0] = s * np.power(F[:, np.newaxis], beta)
        sig[:, :, 1, 0] = alpha * rho * s
        sig[:, :, 1, 1] = alpha * np.sqrt(1.0 - rho ** 2) * s
        return sig

    def fdisc(x):
        [F, s] = x
        return np.zeros((len(F), len(s)))

    n_steps = 2000
    dt = T[-1] / n_steps

    P = np.zeros((len(K), len(T)))

    for i in range(len(K)):
        
        f = lambda x : np.where(x < K[i], 0.0, x - K[i])

        def fpay(x):
            [F, s] = x
            return np.full((len(F), len(s)), f(F)[:, np.newaxis])

        g = fd_option(fpay, fmu, fsig, fdisc, [F, sig], dt, n_steps)

        for j in range(len(T)):

            k = int(np.rint((T[j] / dt)))

            P[i, j] = g[n_steps - k, 20, 20]

    return P

##############################################################################################################

g = None
work_a = None
work_ss = None
init = None

def sabr_prices_fd2(alpha, beta, rho, sig0, f0, mu, K, T, bnds, nx, ny, n_steps):

    P = np.zeros((len(K), len(T)))

    if sig0 == 0:
        for i in range(len(K)):
            if f0 > K[i]:
                P[i, :] = f0 - K[i]
    else:

        global g, work_a, work_ss, init

        if g is None:
            g = np.empty((n_steps + 1, nx, ny))
        if work_a is None:
            work_a = np.empty((nx, ny, nx, ny))
        if work_ss is None:
            work_ss = np.zeros((nx, ny, 2, 2))
        if init is None:
            init = np.empty((nx, ny))


        a = bnds[0]
        b = bnds[1]
        F = np.linspace(a, b, nx)
        dx = (b - a) / (nx - 1)

        # underlying vol range
        a = bnds[2]
        b = bnds[3]
        sig = np.linspace(a, b, ny)
        dy = (b - a) / (ny - 1)
        # Finite difference algorithm (new)

        #s = np.zeros((len(F), len(sig), 2, 2))
        ss = work_ss
        fb = np.power(F[:, np.newaxis], beta)
        ss[:, :, 0, 0] = (sig * fb) ** 2
        ss[:, :, 1, 0] = alpha * rho * (sig ** 2) * fb
        ss[:, :, 0, 1] = ss[:, :, 1, 0]
        ss[:, :, 1, 1] = (alpha * sig) ** 2
    
        dt = T[-1] / n_steps

        init[:] = 0
        ix = int((f0 - F[0]) / dx)
        iy = int((sig0 - sig[0]) / dy)
        init[ix, iy] = 1

        fd_solve_fwd_2d(init, None, ss, F, sig, n_steps, dt, work_a=work_a, out=g)

        for i in range(len(K)):        
            f = lambda x : np.where(x < K[i], 0.0, x - K[i])
            for j in range(len(T)):
                k = int(np.rint((T[j] / dt)))
                P[i, j] = np.sum(np.sum(g[k], axis=1) * f(F)) / np.sum(g[k])
                #P[i, j] = bs_eur_call_ivol(P[i, j], K[i], f0, 0.0, 0.0, T[j], sig0)

    return P

##############################################################################################################

alpha = 0.2
beta = 0.5
rho = 0.4
sig0 = 0.2

f0 = 1
mu = None

a = 0.5
b = 1.5
K = np.linspace(a, b, 21)

T = 0.25
#t = np.array([1/12, 1/6, 0.25, 1/3, 0.5, 1, 2])
t = np.array([T])

n_steps = 60
nx = 41
ny = 41
bnds = (0, 4, 0, 1)

#dt = T / n_steps

#np.random.seed(0)
#n_paths = 62500
#Z1 = gaussian(n_paths, n_steps)
#Z2 = gaussian(n_paths, n_steps)
#Z1[np.abs(Z1) > 6.0] = 0.0
#Z2[np.abs(Z2) > 6.0] = 0.0

P = sabr_prices_fd2(alpha, beta, rho, sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
#P = P * (1 + 0.01 * np.random.randn())
#P = sabr_prices_fd(alpha, beta, rho, sig0, f0, mu, K, t)
#P = sabr_prices_mc(alpha, beta, rho, sig0, f0, mu, K, t, Z1, Z2, dt)

print(np.transpose(P,[1,0]))

#n_paths = 100000
#n_steps = 16
#dt = T / n_steps

#Z1 = gaussian(n_paths, n_steps)
#Z2 = gaussian(n_paths, n_steps)
#Z1[np.abs(Z1) > 6.0] = 0.0
#Z2[np.abs(Z2) > 6.0] = 0.0

#def resids(x):
#    p = sabr_prices_mc(x[0], x[1], x[2], sig0, f0, mu, K, t, Z1, Z2, dt)
#    return (p / P - 1).flatten()

#result = least_squares(resids, [0.8, 0.05, 0.5], bounds=([0.0, 0.0, -1.0], [1.0, 1.0, 1.0]), max_nfev=4000)
#print("MC result: ", result.x)

#r = resids([alpha, beta, rho])
#r2 = resids(result.x)
#print("MC: ", np.sum(r ** 2), np.sum(r2 ** 2))

#def resids_fd(x):
#    p = sabr_prices_fd(x[0], x[1], x[2], sig0, f0, mu, K, t)
#    return (p - P).flatten()

#f_loss = lambda x : np.array([x ** 2, 2 * x, np.full(len(x), 2)])
f_loss = lambda x : np.array([np.sqrt(x), 0.5 / np.sqrt(x), -0.25 / np.sqrt(x)**3])

def resids_fd2(x):
    p = sabr_prices_fd2(x[0], beta, x[1], sig0, f0, mu, K, t, bnds, nx, ny, n_steps)
    return (p - P).flatten()
    #return np.log(p / P).flatten()

def fun_f2(x):
    r = resids_fd2(x)
    return 10000000 * (0.5 * np.sqrt(np.sum(r ** 2)))
    #return np.max(r ** 2)

#result = least_squares(resids_fd2, [0.0,0.0,0.001])
#result = least_squares(resids_fd2, [1.0, 0, 0.001], bounds=([0.0, -1.0, 0.0], [2.0, 1.0, 1.0]))
result = minimize(fun_f2, [0.5, 0.5], bounds=((0,1),(-1,1)), tol=1e-10)
#result = brute(fun_f2, ((0, 1),(-1, 1)), Ns=20)
print("FD result: ", result)
print()

r = resids_fd2([alpha, rho, sig0])
r2 = resids_fd2(result.x)
print("FD: ", 0.5 * np.sum(r ** 2), 0.5 * np.sum(r2 ** 2))

ax = np.linspace(0, 1, 21)
ay = np.linspace(-1, 1, 21)
z = np.zeros((len(ax), len(ay)))

for i in range(len(ax)):
    for j in range(len(ay)):
        z[i, j] = fun_f2((ax[i], ay[j]))

x, y = np.meshgrid(ax, ay, indexing='ij')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
plt.contour(x, y, z, 40)
plt.show()
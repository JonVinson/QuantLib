import numpy as np
import pyodbc as db

from Calibrator import Calibrator
from FDDiffusionModels import HestonModel

from HestonPriceDist import *

##############################################################################################################
def get_prices():
    conn = db.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=ELLA;DATABASE=MARKETDATA;TRUSTED_CONNECTION=YES')
    cur = conn.cursor()
    rows = cur.execute("SELECT (Bid + Ask) / 2 FROM OptionData WHERE Underlying='SPY' AND Type='CALL' AND Expiration = '3/20/2020'" \
       + " AND QuoteDate='12/31/2019' AND Strike BETWEEN 50 AND 420 AND CONVERT(INT,Strike) % 10 = 0  ORDER BY Strike").fetchall()
    return np.array(rows)
##############################################################################################################

mu = 0.03
theta = 0.1228
kappa = 0.1
xi = 0.2
rho = -0.5

sig0 = 0.1228

f0 = 321.85

a = 50
b = 420
K = np.linspace(a, b, 38)

T = 7.0 / 24

n_steps = 60
nx = 41
ny = 41
bnds = (0, 500, 0, 1)

P = get_prices(); # sql prices

G = np.abs(d2(P))
G = G / np.sum(G)

pBnds = ((-1, 1), (0, 1), (0, 1))
varParams = (mu, theta, xi)
varIndex = [0, 1, 3] # looking for xi and rho
fixParams = [kappa, rho]

model = HestonModel()

cal = Calibrator()

cal.SetModel(model)
cal.SetParameters(varParams, pBnds, varIndex, fixParams)
cal.SetLattice(bnds, [nx, ny], T, n_steps)
cal.SetDiffusionStart([f0, sig0])
cal.SetDistribution(G, K)

result = cal.GetResult()

print("FD result: ", result)

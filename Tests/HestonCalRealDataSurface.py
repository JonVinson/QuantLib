import numpy as np
import pyodbc as db
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from Calibrator import Calibrator
from FDDiffusionModels import HestonModel

from HestonPriceDist import *

##############################################################################################################

def get_prices_type_date(symbol, type, quoteDate1, quoteDate2, expiration):
    conn = db.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=ELLA;DATABASE=MARKETDATA;TRUSTED_CONNECTION=YES')
    cur = conn.cursor()
    sql = "SELECT Strike, AVG((Bid + Ask) / 2) FROM OptionData WHERE Underlying=? AND Type=? AND QuoteDate BETWEEN ? AND ? AND Expiration=? GROUP BY Strike"
    rows = cur.execute(sql, symbol, type, quoteDate1, quoteDate2, expiration).fetchall()
    return dict(rows)

##############################################################################################################

def get_prices_date(symbol, quoteDate1, quoteDate2, expiration):
    conn = db.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=ELLA;DATABASE=MARKETDATA;TRUSTED_CONNECTION=YES')
    cur = conn.cursor()
    sql = "SELECT Strike, AVG((Bid + Ask) / 2) FROM OptionData WHERE Underlying=? AND QuoteDate BETWEEN ? AND ? AND Expiration=? GROUP BY Strike"
    rows = cur.execute(sql, symbol, quoteDate1, quoteDate2, expiration).fetchall()
    return dict(rows)

def get_prices_date2(symbol, quoteDate1, quoteDate2, expiration):
    conn = db.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=ELLA;DATABASE=MARKETDATA;TRUSTED_CONNECTION=YES')
    cur = conn.cursor()
    sql = "SELECT Strike, AVG((Bid + Ask) / 2) FROM OptionData WHERE Underlying=?" \
        + " AND QuoteDate BETWEEN ? AND ? AND Expiration=?" \
        + " AND ((Type = 'call' AND Strike > 3300) OR (Type = 'put' AND Strike <= 3300))" \
        + " GROUP BY Strike"
    rows = cur.execute(sql, symbol, quoteDate1, quoteDate2, expiration).fetchall()
    return dict(rows)

##############################################################################################################

def get_prices(strikes, expDates):
    symbol = 'SPX'
    type = 'call'
    quoteDate1 = '12/24/2019'
    quoteDate2 = '12/31/2019'
    result = []
    for date in expDates:
        #prices = get_prices_type_date(symbol, type, quoteDate1, quoteDate2, date)
        prices = get_prices_date2(symbol, quoteDate1, quoteDate2, date)
        result.append([prices.get(k) for k in strikes])
    return np.array(result)

##############################################################################################################


sig0 = 0.1228 # est initial vol
f0 = 3230.78 # initial stock price

mu = 0.0155 # stock drift
theta = sig0 # long term vol
kappa = 0.1 # vol reversion rate
xi = 0.5 # vol of vol
rho = 0.0 # correlation

a = 2000
b = 3700
strikes = np.linspace(a, b, 35)

expDates = ['3/20/2020']

times = np.array([0.25, 0.5, 0.75])
n_steps = 60

nx = 41
ny = 41
bnds = (0, 500, 0, 1)

prices = get_prices(strikes, expDates) # sql prices

#plt.plot(strikes, prices[0]);
#plt.plot(strikes, prices[1]);
#plt.show()
#exit()

#prices = (prices[:,:-1] + prices[:,1:]) / 2
#strikes = (strikes[:-1] + strikes[1:]) / 2

dist = np.abs(diff2(prices, 1))

#dist = np.zeros(np.shape(prices))

#for i in range(len(expDates)):
#    spl = UnivariateSpline(strikes, prices[i], s=0.00001*len(strikes)).derivative(2)
#    dist[i] = spl(strikes)

dist = dist / np.sum(dist, axis=1, keepdims=True)

plt.plot(strikes, dist[0], label='Mar')
#plt.plot(strikes, dist[1], label='Jun')
#plt.plot(strikes, dist[2], label='Sep')
plt.legend()
plt.show()
exit()

pBnds = ((0, 10), (0, 10), (0, 10), (-1, 1))
varParams = (theta, kappa, xi, rho)
varIndex = [1, 2, 3, 4] # looking for xi and rho
fixParams = [mu]

model = HestonModel()

cal = Calibrator()

cal.SetModel(model)
cal.SetParameters(varParams, pBnds, varIndex, fixParams)
cal.SetLattice(bnds, [nx, ny], times[-1], n_steps)
cal.SetDiffusionStart([f0, sig0])
cal.SetDistribution(dist, strikes, times)

result = cal.GetResult()

print("FD result: ", result)

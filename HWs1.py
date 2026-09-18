#import packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sklearn as skl

#smoke test
print('ok', len(yf.Ticker('AAPL').history(period='5d')), 'days pulled')

#Data collection

tsm = yf.download('TSM', period='1y',interval='1d')['Close'].dropna()
spy = yf.download('SPY', period='1y',interval='1d')['Close'].dropna()

#most recent returns
print('last returns of the year:', tsm.iloc[-1,0], spy.iloc[-1,0])

#returns from prev year
rets_tsm =((tsm.iloc[-1,0] - tsm.iloc[0,0]) / tsm.iloc[0,0]) * 100
rets_spy = ((spy.iloc[-1,0] - spy.iloc[0,0]) / spy.iloc[0,0]) * 100

print('returns of TSM:', rets_tsm, '%')
print('returns of SPY:', rets_spy, '%')

#annualized volatility
an_vol_tsm = tsm.pct_change().std() * np.sqrt(252)
an_vol_spy = spy.pct_change().std() * np.sqrt(252)

#rebased returns
rebased_tsm = tsm / tsm.iloc[0,0] * 100
rebased_spy = spy / spy.iloc[0,0] * 100


#largest single move
max_tsm = tsm.pct_change().max().item()
max_tsm_date = tsm.pct_change().idxmax().item()
print('largest single move of TSM:', max_tsm, 'date:', max_tsm_date)

plt.plot()
plt.plot(rebased_tsm, label='TSM')
plt.plot(rebased_spy, label='SPY')

plt.title('TSM vs SPY rebased to 100')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


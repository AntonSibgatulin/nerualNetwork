from pandas_datareader import data as pdr

#data = pdr.get_data_yahoo("GBPUSD", start="2014-01-01", end="2016-01-01")

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt




def CCI(time_period,df):
    multiplier = 0.015
    #df = pd.read_csv('data.csv')

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=time_period).mean()

    mad = np.abs(tp - sma).rolling(window=time_period).mean()
    cci = (tp - sma) / (multiplier * mad)
    return cci

def SMA(time_period,tp):
    return tp['Close'].rolling(window=time_period).mean()

def stochastic_oscillator(prices: pd.DataFrame, k_period: int, d_period: int) -> pd.DataFrame:
    lows = prices["Low"].rolling(window=k_period).min()
    highs = prices["High"].rolling(window=k_period).max()
    k = (prices["Close"] - lows) / (highs - lows) * 100

    d = k.rolling(window=d_period).mean()

    return pd.DataFrame({"%K": k, "%D": d})



def RSI(period,df):
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df = df.fillna(df['RSI'].mean())
    return df['RSI']


def MACDF(data, n_fast, n_slow):
    """
    Расчет индикатора MACD.
    :param data: pandas.DataFrame или pandas.Series.
    :param n_fast: int. Количество периодов для быстрой скользящей средней.
    :param n_slow: int. Количество периодов для медленной скользящей средней.
    :return: pandas.DataFrame or pandas.Series. MACD, быстрый EMA, медленный EMA
    """
    EMA_fast = data.ewm(span=n_fast, min_periods=n_fast).mean()
    EMA_slow = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = EMA_fast - EMA_slow
    MACD_signal = MACD.ewm(span=9, min_periods=9).mean()
    MACD_diff = MACD - MACD_signal
    #return MACD, MACD_signal, MACD_diff
    return MACD_diff


def calculate_macd(df, fast=12, slow=26, signal=9):
    # Calculate fast and slow moving averages
    ema_fast = df['Close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['Close'].ewm(span=slow, min_periods=slow).mean()

    # Calculate MACD line
    macd = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()

    # Calculate histogram
    histogram = macd - signal_line

    # Add columns to the data frame
    df['macd'] = macd
    df['signal'] = signal_line
    df['histogram'] = histogram

    return df

'''
cci = CCI(20,prices)
data = stochastic_oscillator(prices, k_period=14, d_period=3)
rsi = RSI(14,prices)
macd=MACD(prices,12,26)
'''
'''


plt.title("LINE GRAPH")
plt.plot(cci,color = "red")
plt.show()



plt.title("LINE GRAPH")
#plt.plot(data["%K"],color = "blue")
#plt.plot(cci,color = "red")
plt.plot(macd[2],color="green")
plt.show()



import yfinance as yf
stock_list = ['0005.HK']
print('stock_list:', stock_list)
data = yf.download(stock_list, start="2015-01-01", end="2020-02-21")
print('data fields downloaded:', set(data.columns.get_level_values(0)))
data.head()
'''














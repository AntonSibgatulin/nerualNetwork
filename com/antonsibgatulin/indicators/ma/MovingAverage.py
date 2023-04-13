# Moving Averages Code

# Load the necessary packages and modules
from pandas import data as pdr
import matplotlib.pyplot as plt
import yfinance
import pandas as pd

# Simple Moving Average 
def SMA(data, ndays): 
 SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
 data = data.join(SMA) 
 return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
 EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data
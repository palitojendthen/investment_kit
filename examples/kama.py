#!/usr/bin/env python

#load library
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import sys
!git clone https://github.com/palitojendthen/investment_kit
%cd investment_kit/src
%run technical_indicator.py
import technical_indicator

# retrieve data
start_date = '2022-01-01'; end_date = '2024-11-01'
df = yf.download(tickers='BTC-USD', threads=True, start=start_date, end=end_date)
df.columns = df.columns.str.lower().str.replace(' ','_')

# apply kaufman adaptive moving average indicator
kama = technical_indicator.kama(df['close'], return_df=True)
kama['trigger'] = np.where((kama['kama'] > kama['kama'].shift(1)), 1, 0)
kama['buy'] = np.where(kama['trigger'] == 1, kama['kama'], np.nan)
kama['sell'] = np.where(kama['trigger'] == 0, kama['kama'], np.nan)
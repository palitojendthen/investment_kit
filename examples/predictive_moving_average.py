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

# apply predictive moving average indicator
pma = technical_indicator.predictive_moving_average(df['close'], return_df=True)
pma['trigger'] = np.where((pma['series'] > pma['series'].shift(1)), 1, 0)
pma['buy'] = np.where(pma['trigger'] == 1, pma['series'], np.nan)
pma['sell'] = np.where(pma['trigger'] == 0, pma['series'], np.nan)
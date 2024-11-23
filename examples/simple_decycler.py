#!/usr/bin/env python

#load library
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import sys
path = r'path_to_cloned_repository'
sys.path.insert(0, path)
import technical_indicator

# retrieve data
start_date = '2022-01-01'; end_date = '2024-11-01'
df = yf.download(tickers='BTC-USD', threads=True, start=start_date, end=end_date)
df.columns = df.columns.str.lower().str.replace(' ','_')

# apply simple decycler indicator
decycler = technical_indicator.simple_decycler(df['close'], hyst_percentage=1, return_df=True)
decycler['trigger'] = np.where((decycler['decycler'] > decycler['decycler'].shift(1)), 1, 0)
decycler['buy'] = np.where(decycler['trigger'] == 1, decycler['decycler'], np.nan)
decycler['sell'] = np.where(decycler['trigger'] == 0, decycler['decycler'], np.nan)

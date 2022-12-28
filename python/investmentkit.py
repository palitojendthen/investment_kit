
"""
Library
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import scipy.stats
import statsmodels.stats.moment_helpers as mh
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import inv
import plotly.express as px
import yfinance as yf



"""
Technical analysis indicator
"""
# Exponential Moving Average
def ema(x, periods = 14):
    """
    computes exponential moving average
    of given time-series data
    referece: https://www.investopedia.com/terms/e/ema.asp
    """
    n = len(x)
    alpha = 2/(periods+1)
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    else:
        sma_ = x.rolling(window = periods).mean()
        ema = pd.DataFrame({'values':np.nan}, index = x.index)
        for i in range(periods, len(x)):
            ema['values'][i] = alpha*x[i]+(1-alpha)*sma_[i]
    return pd.Series(ema['values'])

# Weighted Moving Average
def wma(x, periods = 14):
    """
    computes weighted moving average,
    of given time-series data
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/wma
    """
    n = len(x)
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    else:
        w = np.arange(1, periods+1)
        w_sum = w.sum()
        weights = w/w_sum
        wma = x.rolling(window = periods).apply(lambda y: np.dot(y, weights), raw = True)    
    return wma

# Hull Moving Average
def hma(x, periods = 14):
    """
    computes hull moving average,
    of given time-series data,
    an improvement to fast and smooth moving average 
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average
    """
    n = len(x)
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    else:
        wma1 = wma(x, periods = int(periods/2))
        wma2 = wma(x, periods = periods)
        hma_ = (2*wma1) - wma2
        hma = wma(hma_, periods = int(np.sqrt(periods)))    
    return hma

# Stochastic Oscillator
def stochastic(close, high, low, periods = 14, return_d = False, smooth = 3):
    """
    computes stochastic oscillator,
    of given time-series data,
    as an technical analysis indicator aim to identify momentum, 
    and identify overbought or oversold area
    reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    n = len(close)
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    else:
        low_ = low.rolling(window = periods).min()
        high_ = high.rolling(window = periods).max()
        stoch = pd.DataFrame({'%k':np.nan}, index = close.index)
        for i in range(periods, n):
            stoch['%k'][i] = ((close[i]-low_[i])/(high_[i]-low_[i]))*100
        if return_d:
            stoch['%d'] = stoch['%k'].rolling(window = smooth).mean()
            return stoch
        return pd.Series(stoch['%k'])
    
# Relative Strength Index
def rsi(x, periods = 14, return_df = None):
    """
    computes relative strength index value,
    of a given time-series data,
    as techincal analysis aim to identify overbought or oversold area,
    reference: https://www.investopedia.com/terms/r/rsi.asp
    """
    n = len(x)
    df = pd.DataFrame({'value':x.copy()}, index = x.index)
    df['diff'] = x.diff()
    df['gain'] = np.where(df['diff'] > 0, df['diff'], 0)
    df['loss'] = np.where(df['diff'] < 0, df['diff'], 0)
    df['avg_gain'] = df['gain'].ewm(com = periods-1, adjust = False).mean()
    df['avg_loss'] = df['loss'].ewm(com = periods-1, adjust = False).mean().abs()
    df['rs'] = df['avg_gain']/df['avg_loss']
    df['rsi'] = (100 - (100/(1+df['rs'])))
    if return_df is not None:
        return df
    else:
        return pd.Series(df['rsi'])


 


# High-Pass Filter
def high_pass_filter(src, hp_period = 48, length = 10):
    """
    high-pass filter i.e. retain a high-frequency components from price data
    return series of a  high-frequency data,
    """
    n = len(src)
    alpha1 = 0.00
    hp = [0.00]*n
    pi = 2*math.asin(1)
    
    alpha1 = ((math.cos(.707*2*pi/hp_period))+(math.sin(.707*2*pi/hp_period)-1))/(math.cos(.707*2*pi/hp_period))
    
    for i in range(2, n):
        hp[0] = 0
        hp[i] = (1-alpha1/2)*(1-alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-alpha1)*hp[i-1]-(1-alpha1)*(1-alpha1)*hp[i-2]

    return pd.Series(hp)

# Ehlers Simple Decycler
def simple_decycler(src, hp_period = 89, n_output = 150, show_hysteresis = True):
    """
    technical analysis indicator originated by John F. Ehlers
    by subtracting high-frequency components from price data, 
    while retain the low-frequency components of price data i.e. trends are kept intact with little to no lag
    return trend, including a hyteresis band
    """
    
    # length
    n = len(src)
    
    # Variable
    alpha1 = 0.00
    hp = [0.00]*n
    hysteresis_up = [0.00]*n
    hysteresis_down = [0.00]*n
    decycler = [0.00]*n
    pi = 2*np.arcsin(1)
    alpha1 = (np.cos(.707*2*pi/hp_period)+np.sin(.707*2*pi/hp_period)-1)/np.cos(.707*2*pi/hp_period)
    
    # simple decycler
    for i in range(1, n):
        
        # high-pass filter
        hp[0] = 0
        hp[i] = (1-alpha1/2)*(1-alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-alpha1)*hp[i-1]-(1-alpha1)*(1-alpha1)*hp[i-2]
        
        # decycler
        decycler[i] = src[i]-hp[i]
        hysteresis_up[i] = decycler[i]*(1+(.5/100))
        hysteresis_down[i] = decycler[i]*(1-(.5/100))

    
    # return
    decycler = decycler[n_output:]
    src = src[n_output:]
    
    # options
    if show_hysteresis:
        hysteresis_up = hysteresis_up[n_output:]
        hysteresis_down = hysteresis_down[n_output:]
        return [decycler, src, hysteresis_up, hysteresis_down]
    else:
        return[decycler, src]




## Portfolio Management ##

# Rate of Return
def rate_of_return(x):
    """
    compute rate of returns of a time series data
    """
    ret = ((x[:-1].values / x[1:]).values - 1).round(2)
    return np.append(np.nan, ret)


# Compounding Returns
def compound(r):
    """
    compute compounding or geometric mean returns 
    for returns series
    """
    return np.expm1(np.log1p(r).sum())

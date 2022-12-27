
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
def ema(x, periods = 3, alpha = .5):
    """
    computes exponential moving average
    of given time-series data
    """
    sma_ = x.rolling(window = periods).mean()
    ema = pd.DataFrame({'values':np.nan}, index = x.index)

    for i in range(periods, len(x)):
        ema['values'][i] = alpha*x[i]+(1-alpha)*sma_[i]
    
    return pd.Series(ema['values'])




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

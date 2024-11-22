#!/usr/bin/env python

"""portfolio_management.py: a collection of tools use to support within portfolio management/asset allocation"""
__author__     = "Palito J. Endthen"
__version__    = "1.0.0"
__maintainer__ = "Palito J. Endthen"
__email__      = "palitoj.endthen@outlook.com"
__license__    = "GPL-3.0"
__status__     = "Prototype"


# Library
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.stats.moment_helpers as mh
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import inv
# import yfinance as yf
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = 'iframe'


# Function
def rate_of_return(x):
    """
    return rate of returns,
    on a given time-series data
    params:
    @x: series, time-series input data
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.rate_of_return(df)
    """
    ret = ((x[1:].values/x[:-1]).values-1).round(2)
    return np.append(np.nan, ret)

def compound(r):
    """
    return compounding or geometric mean returns
    on a given percentage change of a time-series data
    params:
    @r: series, returns over time, of a time-series input data
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.compound(df.pct_change())
    """
    return np.expm1(np.log1p(r).sum())

def semi_deviation(r):
    """
    return semi deviation,
    i.e. deviation from returns that below average or below zero
    params:
    @r: series, returns over time, of a time-series input data
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.semi_deviation(df.pct_change()) 
    """
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def skewness(r):
    """
    return skewness,
    alternative to scipy.stats.skew
    params:
    @r: series, returns over time, of a time-series input data
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.skewness(df.pct_change())
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    return kurtosis,
    alternative to scipy.stats.kurtosis
    params:
    @r: series, returns over time, of a time-seires input data
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.kurtosis(df.pct_change())
    """
    demeaned_r = r-r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = .01):
    """
    return applies jarque-bera test to determine if a series is normaly distributed or not,
    test is applied at the 1% level by default,
    true if the hypothesis of normality is accepted, false otherwise,
    params:
    @r: series, returns over time, of a time-series input data
    @level: float, levels of confidence
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.is_normal(df.pct_change())
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    """
    return historic value-at-risk,
    e.g. means at 95% level of confidence the risk associated,
    estimated price performance would drop at specified level
    params:
    @r: series, returns over time, of a time-series input data
    @level: integer, confidence level
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> portfolio_management.var_historic(df.pct_change().dropna())
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("expected r to be series or data frame")

def annualize_rets(r, periods_per_year=12):
    """
    return annualized set of returns,
    params:
    @r: series, returns over time, of a time-series input data
    @periods_per_year: integer, number of n per periods
    example:
    >>> portfolio_management.annualize_rets(df['close'].pct_change().dropna())
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

def annualize_vol(r, periods_per_year = 12):
    """
    return annualized volatility set of returns,
    params:
    @r: series, returns over time, of a time-series input data
    @periods_per_year: integer, number of n per periods
    example:
    >>> portfolio_management.annualize_vol(df['close'].pct_change().dropna())
    """
    return r.std()*(periods_per_year**.5)

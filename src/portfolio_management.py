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
import statsmodels.api as sm
import math
import scipy.stats
import statsmodels.stats.moment_helpers as mh
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
    @x: time-series input data
    """
    ret = ((x[:-1].values / x[1:]).values - 1).round(2)
    return np.append(np.nan, ret)

def compound(r):
    """
    return compounding or geometric mean returns
    on a given percentage change/returns time-series data
    params:
    @r: returns time-series data
    """
    return np.expm1(np.log1p(r).sum())

def semi_deviation(r):
    """
    return semi deviation,
    i.e. deviation from returns that below average or below zero
    params:
    @r: returns time-series data
    """
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def skewness(r):
    """
    return skewness of the supplied series data,
    i.e. alternative to scipy.stats.skew
    returns a float or a series
    params:
    @r: returns time-series data
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    return kurtosis value of the supplied series data,
    i.e. alternative to scipy.stats.kurtosis
    returns a float or a series
    params:
    @r: returns time-series data
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = .01):
    """
    return applies jarque-bera test to determine if a series is normaly distributed or not
    test is applied at the 1% level by default
    true if the hypothesis of normality is accepted, false otherwise
    params:
    @r: returns time-series data
    @level: confidence levels
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    """
    return historic value-at-risk,
    which means 95% i.e. level of confidence, that at worst
    the estimated price performance would drop at specified level
    @r: returns time-series data
    @level: confidence levels
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
    @r: returns time-series data
    @periods per year: default to 12 months
    
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

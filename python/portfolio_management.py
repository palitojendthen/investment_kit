
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
Portfolio Management/Optimization
"""
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

# Historic Value-at-Risk
def var_historic(r, level = 5):
    """
    computes historic value at risk (var),
    which means 95% i.e. level of confidence, that at worst
    the estimated price performance would drop at specified level
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("expected r to be series or data frame")

# Annualize Returns
def annualize_rets(r, periods_per_year=12):
    """
    compute the annualized set of returns,
    periods per year default to 12 - monthly
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

# Annualize Volatility
def annualize_vol(r, periods_per_year = 12):
    """
    compute the annualized volatility set of returns,
    periods per year default to 12 - monthly
    """
    return r.std()*(periods_per_year**.5)

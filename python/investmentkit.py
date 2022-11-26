
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
import plotly.express as px
import yfinance as yf



def rate_of_return(x):
    """
    compute rate of returns of a time series data
    """
    ret = ((x[:-1].values / x[1:]).values - 1).round(2)
    return np.append(np.nan, ret)

def compound(r):
    """
    compute compounding or geometric mean returns 
    for returns series
    """
    return np.expm1(np.log1p(r).sum())
#!/usr/bin/env python

"""technical_indicator.py: a collection of technical analysis/indicator used to support an applied trading strategy/algorithmic trading"""
__author__     = "Palito Endthen"
__version__    = "1.0.0"
__maintainer__ = "Palito Endthen"
__email__      = "palitoj.endthen@outlook.com"
__license__    = "GPL-3.0"
__status__     = "Prototype"


# Library
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
# import yfinance as yf


# function
def fibonacci(n):
    """
    return sequence number generated based on fibonacci sequence,
    given the number of n
    params:
    @n: integer,number of expected output data

    example:
    >>> technical_indicator.fibonacci(5)
    [0, 1, 1, 2, 3, 5]
    """
    f = [0,1]
    for i in range(2, n + 1):
        f.append(f[i - 1] + f[i - 2])        
    return f

def fibonacci_retracement_sr(_bottom, _top):
    """
    return retracement level of a given price range,
    i.e. 23.6%, 38.2%, 50%, 61.8%, and 100%,
    based on the fibonacci retracement with specified bottom and top value,
    which aim identified as support/resistance level,
    reference: https://www.investopedia.com/terms/f/fibonacciretracement.asp    
    params:
    @_bottom: number, bottom price level
    @_top: number, peak price level
    example:
    >>> technical_indicator.fibonacci_retracement_sr(10, 15.8)
    array([15.8   , 14.4312, 13.5844, 12.9   , 12.2156, 11.2412, 15.742 ])
    """
    if _bottom > _top:
        raise ValueError("Bottom can't be greater than Top value")

    _range = _top - _bottom
    _ratio =  pd.Series([0, 23.6, 38.2, 50, 61.8, 78.6, 1])/100
    _level = []
    
    for i in _ratio:
        _results = _bottom+_range*(1-i)
        _level.append(_results)
    
    return np.array(_level)

def ema(src, periods = 14):
    """
    technical analysis indicator:
    return exponential moving average,
    on a given time-series data
    referece: https://www.investopedia.com/terms/e/ema.asp
    params:
    @src: series, time-series input data
    @periods: integer, n lookback period
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(close, columns = ['close'])
    >>> technical_indicator.ema(df['close'])[-5:]
    15    20.961905
    16    21.033333
    17    22.409524
    18    22.338095
    19    23.119048
    """
    src = src.dropna()
    n = len(src)
    alpha = 2/(periods+1)
    
    if n < periods:
        raise ValueError("Periods can't be greater than data length")
    
    _sma = src.rolling(window = periods).mean()
    _ema = pd.DataFrame({'values':np.nan}, index = src.index)
    
    for i in range(periods, n):
        _ema['values'][i] = alpha*src[i]+(1-alpha)*_sma[i]
    
    return pd.Series(_ema['values'])

def wma(src, periods = 14):
    """
    technical analysis indicator:
    return weighted moving average,
    on a given time-series data
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/wma
    params:
    @src: time-series input data
    @periods: n lookback period
    """
    src = src.dropna()
    n = len(src)
    
    if n < periods:
        raise ValueError('Periods cant be greater than data length')
    
    w = np.arange(1, periods+1)
    w_sum = w.sum()
    weights = w/w_sum
    wma = src.rolling(window = periods).apply(lambda y: np.dot(y, weights), raw = True)    
    
    return wma

def hma(src, periods = 14):
    """
    technical analysis indicator:
    return hull moving average,
    on a given time-series data,
    an improvement to fast and smooth moving average
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull-moving-average
    params:
    @src: time-series input data
    @periods: n lookback period
    """
    src = src.dropna()
    n = len(src)
    
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    
    wma1 = wma(src, periods = int(periods/2))
    wma2 = wma(src, periods = periods)
    _hma = (2*wma1)-wma2
    hma = wma(_hma, periods = int(np.sqrt(periods)))    
    
    return hma

def stochastic(close, high, low, periods = 14, smooth = 3, return_df = False, ):
    """
    technical analysis indicator:
    return stochastic oscillator,
    on a given time-series data,
    aim to identify momentum, overbought and oversold area
    reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    params:
    @close: close price time-series data
    @high: high price time-series data
    @low: low price time-series data
    @periods: n lookback period
    @return df: default to false, if true would return as dataframe
    @smooth: smoothing function
    """
    close, high, low = close.dropna(), high.dropna(), low.dropna()
    n = len(close)
    
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    
    _low = low.rolling(window = periods).min()
    _high = high.rolling(window = periods).max()
    _stoch = pd.DataFrame({'%k':np.nan}, index = close.index)
    
    for i in range(periods, n):
        _stoch['%k'][i] = ((close[i]-_low[i])/(_high[i]-_low[i]))*100
    if return_df:
        _stoch['%d'] = _stoch['%k'].rolling(window = smooth).mean()
        return _stoch
    return pd.Series(_stoch['%k'])

def rsi(src, periods = 14, return_df = False):
    """
    techical analysis indicator:
    return relative strength index,
    on a given time-series data,
    aim to identify overbought or oversold area,
    reference: https://www.investopedia.com/terms/r/rsi.asp
    params:
    @src: time-series input data
    @periods: n lookback period
    @return df: default to false, if true would return as dataframe
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')

    df = pd.DataFrame({'value':src.copy()}, index = src.index)
    df['diff'] = src.diff()
    df['gain'] = np.where(df['diff'] > 0, df['diff'], 0)
    df['loss'] = np.where(df['diff'] < 0, df['diff'], 0)
    df['avg_gain'] = df['gain'].ewm(com = periods-1, adjust = False).mean()
    df['avg_loss'] = df['loss'].ewm(com = periods-1, adjust = False).mean().abs()
    df['rs'] = df['avg_gain']/df['avg_loss']
    df['rsi'] = (100 - (100/(1+df['rs'])))
    if return_df is True:
        return df[periods:]
    else:
        return pd.Series(df['rsi'][periods:])

def stochastic_rsi(src, periods = 14, smooth = 3):
    """
    technical analysis indicator:
    return stochastic oscillator value,
    on an input rsi value,
    a momemntum indicator to identify overbought > 0.8,
    and oversold < 0.2,
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/stochrsi
    params:
    @src: time-series input data
    @periods: n lookback period
    @smooth: smoothing function
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')
    
    _rsi = rsi(src, periods = periods)
    _low = _rsi.rolling(window = periods).min()
    _high = _rsi.rolling(window = periods).max()
    _stoch_rsi = pd.DataFrame({'%k':((_rsi-_low)/(_high-_low))*100})
    _stoch_rsi['%d'] = _stoch_rsi['%k'].rolling(window = smooth).mean()

    return _stoch_rsi

def simple_decycler(src, hp_period = 48, hyst_percentage = 5, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers, with aim to identified trend,
    of a given time-series data, by subtracting high-frequency, 
    while retain the low-frequency components of price data,
    trends are kept intact with little to no lag
    reference: https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/E-F/EhlersSimpleDecycler
    params:
    @src: time-series input data
    @hp period: length of a high-pass period e.g. 48, 89, 125
    @return df: default to false, if true would return as dataframe
    """
    src = src.dropna()
    n = len(src)
    
    if n < hp_period:
        raise ValueError('Periods cant be greater than data length')
    
    _df = pd.DataFrame({
        'close':src,
        'hp':0.00,
        'decycler':0.00,
        'hyst_up':0.00,
        'hyst_dn':0.00
    }, index = src.index)
    
    _pi = 2*np.arcsin(1)
    _alpha1 = (np.cos(.707*2*_pi/hp_period)+np.sin(.707*2*_pi/hp_period)-1)/np.cos(.707*2*_pi/hp_period)
    
    for i in range(hp_period, n):
        _df['hp'][i] = (1-_alpha1/2)*(1-_alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-_alpha1)*_df['hp'][i-1]-(1-_alpha1)*(1-_alpha1)*_df['hp'][i-2]
        _df['decycler'][i] = src[i]-_df['hp'][i]
        _df['hyst_up'][i] = _df['decycler'][i]*(1+(hyst_percentage/100))
        _df['hyst_dn'][i] = _df['decycler'][i]*(1-(hyst_percentage/100))
    
    if return_df:
        return _df.iloc[hp_period:, :]
    else:
        return _df['decycler'][hp_period:]

def predictive_moving_average(src, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers,
    by taking difference of 2 lagging line of 7-bars Weighted Moving Average,
    given signal when predict crossing it's trigger
    reference: John F. Ehlers, Rocket Science for Traders pg. 212
    params:
    @src: time-series input data
    @return df: default to false, if true would return as dataframe
    """
    src = src.dropna()
    n = len(src)
    
    _df = pd.DataFrame({
        'close':src,
        'wma1': 0.00,
        'wma2':0.00,
        'predict':0.00,
        'trigger':0.00,
        'series':0.00
    }, index = src.index)
    
    for i in range(7, n):
        _df['wma1'][i] = (7*src[i]+6*src[i-1]+5*src[i-2]+4*src[i-3]+3*src[i-4]+2*src[i-5]+src[i-6])/28
        _df['wma2'][i] = (7*_df['wma1'][i]+6*_df['wma1'][i-1]+5*_df['wma1'][i-2]+4*_df['wma1'][i-3]+3*_df['wma1'][i-4]+2*_df['wma1'][i-5]+_df['wma1'][i-6])/28
        _df['predict'][i] = (2*_df['wma1'][i])-_df['wma2'][i]
        _df['trigger'][i] = (4*_df['predict'][i]+3*_df['predict'][i-1]+2*_df['predict'][i-2]+_df['predict'][i])/10
        if _df['predict'][i] > _df['trigger'][i]:
            _df['series'][i] = _df['predict'][i]
        else:
            _df['series'][i] = _df['trigger'][i]
    
    if return_df:
        return _df.iloc[(7*3):]
    else:
        return _df['series'][(7*3):]

def even_better_sinewave(src, hp_period = 48, return_df = None):
    """
    technical analysis indicator:
    originate by John F. Ehlers, aim to create artificially predictive indicator,
    by transfering cyclic data swings into a sinewave
    referece: John F. Ehlers, Cycle Analytics for Traders pg. 159
    params:
    @src: time-series input data
    @hp_period: length of a high-pass period e.g. 48, 89, 125
    @return df: default to false, if true would return as dataframe
    """
    src = src.dropna()
    n = len(src)
    
    if n < hp_period:
        raise ValueError('Periods cant be greater than data length')

    _df = pd.DataFrame({
        'close': src,
        'hp': 0.00,
        'decycler': 0.00,
        'filt': 0.00,
        'wave': 0.00,
        'pwr': 0.00,
    }, index = src.index)
    
    _pi = 2*np.arcsin(1)
    _alpha1 = (np.cos(.707*2*_pi/hp_period)+np.sin(.707*2*_pi/hp_period)-1)/np.cos(.707*2*_pi/hp_period)

    for i in range(hp_period, n):
        _df['hp'][i] = (1-_alpha1/2)*(1-_alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-_alpha1)*_df['hp'][i-1]-(1-_alpha1)*(1-_alpha1)*_df['hp'][i-2]
        _df['filt'][i] = (7*_df['hp'][i]+6*_df['hp'][i-1]+5*_df['hp'][i-2]+4*_df['hp'][i-3]+3*_df['hp'][i-4]+2*_df['hp'][i-5]+_df['hp'][i])/28
        _df['wave'][i] = (_df['filt'][i]+_df['filt'][i-1]+_df['filt'][i-2])/3
        _df['pwr'][i] = (_df['filt'][i]*_df['filt'][i]+_df['filt'][i-1]*_df['filt'][i-1]+_df['filt'][i-2]*_df['filt'][i-2])/3
        _df['wave'][i] = _df['wave'][i]/np.sqrt(_df['pwr'][i])

    if return_df:
        return _df.iloc[hp_period:, :]
    else:
        return _df['wave'][hp_period:]

def kama(src, length = 14, fast_length = 2, slow_length = 30, return_df = False):
    """
    technical analysis indicator:
    originated by Perry J. Kaufman,
    an adaptive trendline indicator, that one changes with market conditions,
    with notion of using the fastest trend possible, based on the smallest calculation period,
    for the existing market conditions, by applying an exponential smoothing formula,
    to vary the speed of the trend
    reference: https://corporatefinanceinstitute.com/resources/capital-markets/kaufmans-adaptive-moving-average-kama/
    params:
    @src: time-series input data
    @length: periods that affect efficiency ratio, keep the n below 14, period > 14 will not
        change the value, only make them smaller
    @fast length: the fastest values represent the range of calc periods, default to 2
    @slow length: the slowest values represent the range of calc periods, default to 30
    return df: default to false, if true would return as dataframe
    """
    src = src.dropna()
    n = len(src)

    if n < length:
        raise ValueError('Periods cannot be greater than data length')

    fastest = 2/(fast_length+1)
    slowest = 2/(slow_length+1)

    _df = pd.DataFrame({
        'close': src,
        'num': 0.00,
        'delta': 0.00,
        'denom': 0.00,
        'er': 0.00,
        'sc':0.00,
        'kama':0.00
    }, index = src.index)

    for i in range(length, n):
        _df['num'][i] = abs(src[i]-src[i-length])
        _df['delta'][i] = abs(src[i]-src[i-1])
        _df['denom'][length-1:] = np.convolve(_df['delta'], np.ones(length), 'valid')
        _df['er'][i] = _df['num'][i]/_df['denom'][i]
        _df['sc'][i] = math.pow(_df['er'][i]*(fastest-slowest)+slowest, 2)
        _df['kama'][i] = _df['kama'][i-1]+_df['sc'][i]*(_df['close'][i]-_df['kama'][i-1])

    if return_df:
        return _df.iloc[length:, :]
    else:
        return _df['kama'][length:]


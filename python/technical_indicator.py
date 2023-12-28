"""technical_indicator.py: a collection of technical analysis/indicator used to support an applied trading strategy/algorithmic trading"""
__author__     = "Palito Endthen"
__version__    = "1.0.0"
__maintainer__ = "Palito Endthen"
__email__      = "palitoj.endthen@outlook.com"
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
import yfinance as yf


def fibonacci_retracement_sr(_bottom, _top):
    """
    return retracement level of a given price range,
    e.g. 23.6%, 38.2%, 50%, 61.8%, and 100%
    based on the fibonacci retracement
    with specified peak and bottom
    which aim to use support/resistance level among traders,
    reference: https://www.investopedia.com/terms/f/fibonacciretracement.asp
    params:
    @top: identified peak/resistance current level
    @bottom: identified bottom/support current level
    """
    if _bottom > _top:
        raise ValueError('Bottom/Support cant be greater than Top/Peak value')

    _range = _top - _bottom
    _ratio =  pd.Series([0, 23.6, 38.2, 50, 61.8, 78.6, 1])/100
    _level = []
    
    for i in _ratio:
        _results = _bottom+_range*(1-i)
        _level.append(_results)
    
    return pd.Series(_level)

def ema(src, periods = 14):
    """
    technical analysis indicator:
    return exponential moving average,
    on a given time-series data
    referece: https://www.investopedia.com/terms/e/ema.asp
    params:
    @src: time-series input data
    @periods: n lookback period 
    """
    src = src.dropna()
    n = len(src)
    alpha = 2/(periods+1)
    
    if n < periods:
        raise ValueError('Periods cant be greater than data length')
    
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

# Ehlers - Simple Decycler
def simple_decycler(src, hp_period = 89, return_df = False):
    """
    technical analysis indicator originated by John F. Ehlers
    by subtracting high-frequency components from price data, 
    while retain the low-frequency components of price data,
    i.e. trends are kept intact with little to no lag
    return trend, including a hyteresis band
    reference: https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/E-F/EhlersSimpleDecycler
    """
    src = src.dropna()
    n = len(src)
    if n < hp_period:
        raise ValueError('Periods cannot be greater than data length')
    else:
        hp = [0.00]*n
        decycler = [0.00]*n
        hysteresis_up = [0.00]*n
        hysteresis_down = [0.00]*n
        pi = 2*np.arcsin(1)
        alpha1 = (np.cos(.707*2*pi/hp_period)+np.sin(.707*2*pi/hp_period)-1)/np.cos(.707*2*pi/hp_period)

        for i in range(1, n):
            hp[i] = (1-alpha1/2)*(1-alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-alpha1)*hp[i-1]-(1-alpha1)*(1-alpha1)*hp[i-2]
            decycler[i] = src[i]-hp[i]
            hysteresis_up[i] = decycler[i]*(1+(.5/100))
            hysteresis_down[i] = decycler[i]*(1-(.5/100))
        if return_df:
            return pd.DataFrame({'hp':hp[hp_period:],
                                 'price':src[hp_period:],
                                 'decycler':decycler[hp_period:], 
                                 'hysteresis_up':hysteresis_up[hp_period:], 
                                 'hysteresis_down':hysteresis_down[hp_period:]})
        else:
            return pd.Series(decycler[hp_period:])

# Ehlers - Predictive Moving Average
def predictive_moving_average(src, return_df = False):
    """
    technical analysis indicator originated by John F. Ehlers,
    by taking difference of 2 lagging line of 7-bars Weighted Moving Average,
    given signal when predict crossing it's trigger
    reference: John F. Ehlers, Rocket Science for Traders pg. 212
    """
    src = src.dropna()
    n = len(src)
    wma1 = [0.00]*n
    wma2 = [0.00]*n
    predict = [0.00]*n
    trigger = [0.00]*n
    series_ = [0.00]*n
    for i in range(7, n):
        wma1[i] = (7*src[i] + 6*src[i-1] + 5*src[i-2] + 4*src[i-3] + 3*src[i-4] + 2*src[i-5] + src[i-6])/28
        wma2[i] = (7*wma1[i] + 6*wma1[i-1] + 5*wma1[i-2] + 4*wma1[i-3] + 3*wma1[i-4] + 2*wma1[i-5] + wma1[i-6])/28
        predict[i] = (2*wma1[i])-wma2[i]
        trigger[i] = (4*predict[i] + 3*predict[i-1] + 2*predict[i-2] + predict[i])/10
        if predict[i] > trigger[i]:
            series_[i] = predict[i]
        else:
            series_[i] = trigger[i]
    if return_df:
        return pd.DataFrame({'price':src[14:],
                             'predict':predict[14:],
                             'trigger':trigger[14:]})
    else:
        return pd.Series(series_[14:])

# Ehlers - Even Better Sinewave
def even_better_sinewave(src, hp_period = 89, return_df = None):
    """
    technical analysis indicator by John F. Ehlers,
    aims to create artificially predictive indicator,
    by transfering cyclic data swings into a sinewave
    referece: John F. Ehlers, Cycle Analytics for Traders pg. 159
    """
    src = src.dropna()
    n = len(src)
    if n < hp_period:
        raise ValueError('Periods cannot be greater than data length')
    else:
        hp = [0.00]*n
        decycler = [0.00]*n
        filt = [0.00]*n
        wave = [0.00]*n
        pwr = [0.00]*n
        pi = 2*np.arcsin(1)
        alpha1 = (np.cos(.707*2*pi/hp_period)+np.sin(.707*2*pi/hp_period)-1)/np.cos(.707*2*pi/hp_period)

        for i in range(1, n):
            hp[i] = (1-alpha1/2)*(1-alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-alpha1)*hp[i-1]-(1-alpha1)*(1-alpha1)*hp[i-2]
            filt[i] = (7*hp[i] + 6*hp[i-1] + 5*hp[i-2] + 4*hp[i-3] + 3*hp[i-4] + 2*hp[i-5] + hp[i])/28
            wave[i] = (filt[i]+filt[i-1]+filt[i-2])/3
            pwr[i] = (filt[i]*filt[i]+filt[i-1]*filt[i-1]+filt[i-2]*filt[i-2])/3

            wave[i] = wave[i]/np.sqrt(pwr[i])
        
        if return_df:
            return pd.DataFrame({'price': src[hp_period:],
                                'wave': wave[hp_period:]})
        else:
            return pd.Series(wave[hp_period:])

# Kaufman - Adaptive Moving Average
def kama(src, length = 14, fast_length = 2, slow_length = 30, return_df = False):
    """
    technical analysis indicator originated by Perry J. Kaufman,
    an adaptive trendline indicator, with notion of using the fastest trend possible,
    based on the smallest calculation period for the existing market conditions,
    by applying an exponential smoothing formula to vary the speed of the trend,
    reference: https://corporatefinanceinstitute.com/resources/capital-markets/kaufmans-adaptive-moving-average-kama/
    """
    src = src.dropna()
    n = len(src)

    if n < length:
        raise ValueError('Periods cannot be greater than data length')
    else:
        fastest = 2/(fast_length+1)
        slowest = 2/(slow_length+1)
        
        # variable
        num = [0.00]*n
        delta = [0.00]*n
        denom = [0.00]*n
        er = [0.00]*n
        sc = [0.00]*n
        kama = [0.00]*n
        
        for i in range(length, n):
            
            # efficiency ratio
            num[i] = abs(src[i] - src[i-length])
            delta[i] = abs(src[i] - src[i-1])
            denom[length-1:] = np.convolve(delta, np.ones(length), 'valid')
            er[i] = num[i]/denom[i]
            
            # smoothing constant
            sc[i] = math.pow(er[i]*(fastest-slowest)+slowest, 2)
            
            # adaptive moving average
            kama[i] = kama[i-1]+sc[i]*(src[i]-kama[i-1])
        
        if return_df:
            return pd.DataFrame({'price':src[slow_length+length:],
                                 'kama': kama[slow_length+length:]})
        else:
            return pd.Series(kama[slow_length+length:])


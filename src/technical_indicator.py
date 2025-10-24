#!/usr/bin/env python

"""technical_indicator.py: a collection of technical analysis/indicator used to support an applied trading strategy/algorithmic trading"""
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
# import yfinance as yf


# Function
def fibonacci(n):
    """
    return sequence number generated based on fibonacci sequence,
    given the number of n
    params:
    @n: integer, number of expected output data
    example:
    >>> technical_indicator.fibonacci(5)
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

def sma(src, periods=14):
    """
    technical analysis indicator:
    return simple moving average,
    on a given time-series data
    referece: https://www.investopedia.com/terms/s/sma.asp
    params:
    @src: series, time-series input data
    @periods: integer, n lookback period
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> technical_indicator.sma(df['close'])[-5:]
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError("Periods can't be greater than data length")

    _sma = src.rolling(window = periods).mean()

    return pd.Series(_sma)

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
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> technical_indicator.ema(df['close'])[-5:]
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
    @src: series, time-series input data
    @periods: integer, n lookback period
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> technical_indicator.wma(df['close'])
    """
    src = src.dropna()
    n = len(src)
    
    if n < periods:
        raise ValueError("Periods can't be greater than data length")
    
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
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/hull moving-average
    params:
    @src: series, time-series input data
    @periods: integer, n lookback period
    example:
    >>> arr = np.random.randint(10, 30, 20)
    >>> df = pd.DataFrame(arr, columns = ['close'])
    >>> technical_indicator.hma(df['close'])
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

def rma(series, periods=10):
    """
    technical analysis indicator:
    return moving average in RSI,
    an exponentially weighted moving average with alpha = 1/length
    on a given time-series data
    reference: TradingView, https://www.tradingview.com/pine-script-reference/v5/#fun_ta.rma
    params:
    @series: series, time-series input data
    @periods: integer, n loockback period
    example:
    >>> technical_indicator.rma(df['ohlc4'])
    """
    
    alpha = 1/periods
    result = []

    for i in range(len(series)):
        if i < periods:
            result.append(np.nan)
        elif i == periods:
            seed = series.iloc[:periods].mean()
            result.append(seed)
        else:
            prev = result[-1]
            value = alpha*series.iloc[i]+(1-alpha)*prev
            result.append(value)

    return pd.Series(result, index=series.index)

def atr(src, periods = 10):
    """
    technical analysis indicator:
    return average true range
    on a given time-series data
    reference: https://www.investopedia.com/terms/a/atr.asp
    params:
    @src: series, time-series input data
    @periods: integer, n loockback period
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.atr(df, return_df = True)
    """
    src = src.dropna()
    n = len(src)
    
    if n < periods:
        raise ValueError('Periods cant be greater than data length')

    src['hl'] = src['high']-src['low']
    src['hc1'] = src['high']-src['close'].shift(1)
    src['lc1'] = src['low']-src['close'].shift(1)
    src['tr'] = .00

    for i in range(0, len(src)):
        src['tr'][i] = np.max([src['hl'][i], src['hc1'][i], src['lc1'][i]], axis=0)
    
    src['atr'] = src['tr'].rolling(window=periods).mean()
    
    return src[['tr', 'atr']]

def adx(src, periods=10):
    """
    technical analysis indicator:
    return the Average Directional Index,
    aim to determine the strength of a financial security's price trend,
    reference: https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp
    params:
    @src: series, time-series input data
    @periods: integer, n loockback period
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.adx(df,periods=14)
    
    """

    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')

    src['dx_up'] = src['high']-src['high'].shift(1)
    src['dx_down'] = -(src['low']-src['low'].shift(1))
    src['plus_dm'] = np.where(((src['dx_up'] > src['dx_down']) & (src['dx_up'] > 0)), src['dx_up'], 0)
    src['minus_dm'] = np.where(((src['dx_down'] > src['dx_up']) & (src['dx_down'] > 0)), src['dx_down'], 0)
    src['tr'] = atr(src,periods=periods)['tr']
    src['truerange'] = rma(src['tr'], periods=periods)
    src['plus'] = (100*rma(src['plus_dm'],periods=periods)/src['truerange']).fillna(0)
    src['minus'] = (100*rma(src['minus_dm'],periods=periods)/src['truerange']).fillna(0)
    src['adx'] = 100*rma(abs(src['plus']-src['minus'])/np.where((src['plus']+src['minus'])==0, 1, src['plus']+src['minus']), periods=periods)

    return src['adx']

def stochastic(src, close, high, low, periods = 14, smooth = 3, return_df = False):
    """
    technical analysis indicator:
    return stochastic oscillator,
    on a given time-series data,
    aim to identify momentum, overbought and oversold area
    reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    params:
    @close: series, close price of a time-series data
    @high: series, high price of a time-series data
    @low: series, low price of a time-series data
    @periods: integer, n lookback period
    @smooth: integer, smoothing function
    @return_df: boolean, whether to return include input dataframe or result only
    >>> technical_indicator.stochastic(df, 'close', 'high', 'low', return_df=True)
    """
    src = src.dropna()
    close, high, low = src[close], src[high], src[low]
    n = len(close)
    
    if n < periods:
        raise ValueError('Periods cannot be greater than data length')
    
    _low = low.rolling(window = periods).min()
    _high = high.rolling(window = periods).max()
    src['%k'] = .00
    
    for i in range(periods, n):
        src['%k'][i] = ((close[i]-_low[i])/(_high[i]-_low[i]))*100
    if return_df:
        src['%d'] = src['%k'].rolling(window=smooth).mean()
        return src
    return pd.Series(src['%k'])

def rsi(src, col, periods = 14, return_df = False):
    """
    techical analysis indicator:
    return relative strength index,
    on a given time-series data,
    aim to identify overbought or oversold area,
    reference: https://www.investopedia.com/terms/r/rsi.asp
    params:
    @src: series, time-series input data
    @col: strings, input data column
    @periods: integer, n lookback period
    @return_df: boolean, whether to return include input dataframe or result only
    >>> technical_indicator.rsi(df, 'close', return_df=True)
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')

    src['diff'] = src[col].diff()
    src['gain'] = np.where(src['diff'] > 0, src['diff'], 0)
    src['loss'] = np.where(src['diff'] < 0, src['diff'], 0)
    src['avg_gain'] = src['gain'].ewm(com = periods-1, adjust = False).mean()
    src['avg_loss'] = src['loss'].ewm(com = periods-1, adjust = False).mean().abs()
    src['rs'] = src['avg_gain']/src['avg_loss']
    src['rsi'] = (100-(100/(1+src['rs'])))

    if return_df:
        return src[periods:]
    else:
        return pd.Series(src['rsi'][periods:])

def stochastic_rsi(src, col, periods = 14, smooth = 3, return_df = False):
    """
    technical analysis indicator:
    return stochastic oscillator value,
    on an input relative strength index value,
    a momemntum indicator to identify overbought > 0.8,
    and oversold < 0.2,
    reference: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/stochrsi
    params:
    @src: series, time-series input data
    @col: strings, input data column
    @periods: integer, n lookback period
    @smooth: integer, smoothing function
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.stochastic_rsi(df, 'close', return_df = True)
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')
    
    src['rsi'] = rsi(src, col, periods = periods)
    src['low'] = src['rsi'].rolling(window = periods).min()
    src['high'] = src['rsi'].rolling(window = periods).max()
    src['%k'] = ((src['rsi']-src['low'])/(src['high']-src['low']))*100
    src['%d'] = src['%k'].rolling(window=smooth).mean()

    if return_df:
        return src[periods:]
    else:
        return src[['%k', '%d']][periods:]

def simple_decycler(src, hp_period = 48, hyst_percentage = 5, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers, with aim to identified trend,
    of a given time-series data, by subtracting high-frequency component, 
    while retain the low-frequency components of price data,
    trends are kept intact with little to no lag
    reference: https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/E-F/EhlersSimpleDecycler
    params:
    @src: series, time-series input data
    @hp_period: integer, length of a high-pass period e.g. 48, 89, 125
    @hyst_percentage: integer, hysteresis band percentage %
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.simple_decycler(df['close'], hp_period=89, hyst_percentage=3, return_df=True)
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
    @src: series, time-series input data
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.predictive_moving_average(df['close'], return_df=True)
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
        return _df[['predict','trigger','series']][(7*3):]

def even_better_sinewave(src, hp_period = 48, return_df = None):
    """
    technical analysis indicator:
    originate by John F. Ehlers, aim to create artificially predictive indicator,
    by transfering cyclic data swings into a sinewave
    referece: John F. Ehlers, Cycle Analytics for Traders pg. 159
    params:
    @src: series, time-series input data
    @hp_period: integer, length of a high-pass period e.g. 48, 89, 125
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.even_better_sinewave(df['close'], return_df=True) 
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
    @src: series, time-series input data
    @length: integer, periods that affect efficiency ratio, keep the n below 14, period > 14 will not change the value, only make them smaller
    @fast_length: integer, the fastest values represent the range of calc periods, default to 2
    @slow_length: integer, the slowest values represent the range of calc periods, default to 30
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.kama(df['close'], return_df=True)
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

def zero_mean_roofing_filter(src, hp_period = 48, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers, 
    with aim to reducing noise in price data,
    by eleminating wave components with long periods which are perceived as spectral dillation,
    the filter only passes those spectral components whose periods are between 10 and 48 bars,
    the technique noticeably reduces indicator lag and also help assess turning points more accurate
    reference: https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/E-F/EhlersRoofingFilter
    params:
    @src: series, time-series input data
    @hp_period: integer, length of a high-pass period e.g. 48, 89, 125
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.zero_mean_roofing_filter(df['ohlc4'])
    """
    src = src.dropna()
    n = len(src)
    
    if n < hp_period:
        raise ValueError('Periods cant be greater than data length')
        
    _df = pd.DataFrame({
        'close': src,
        'hp': 0.00,
        'filt': 0.00,
        'filt2': 0.00
    }, index = src.index)
    
    _pi = 2*np.arcsin(1)
    a1 = math.exp(-1.414*_pi/10)
    b1 = 2*a1*math.cos(1.414*180/10)
    c2 = b1
    c3 = -a1*a1
    c1 = 1-c2-c3

    _alpha1 = (np.cos(360/hp_period)+np.sin(360/hp_period)-1)/np.cos(360/hp_period)
    
    for i in range(hp_period, n):
        _df['hp'][i] = (1-_alpha1/2)*(_df['close'][i]-_df['close'][i-1])+(1-_alpha1)*_df['hp'][i-1]
        # _df['hp'][i] = (1-_alpha1/2)*(1-_alpha1/2)*(src[i]-2*src[i-1]+src[i-2])+2*(1-_alpha1)*_df['hp'][i-1]-(1-_alpha1)*(1-_alpha1)*_df['hp'][i-2]
        _df['filt'][i] = c1*(_df['hp'][i]+_df['hp'][i-1])/2+c2*_df['filt'][i-1]+c3*_df['filt'][i-2]
        _df['filt2'][i] = (1-_alpha1/2)*(_df['filt'][i]-_df['filt'][i-1])+(1-_alpha1)*_df['filt2'][i-1]
 
    if return_df:
        return _df.iloc[hp_period:, :]
    else:
        return _df[['filt', 'filt2']][hp_period:]

def cyber_cycle(src, lag = 9, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers,
    aim to isolated cycle mode components,
    with smoothing and remove the two-three bar components,
    that detract interpretation of the cyclic signals,
    with finite impulse response
    reference: John F. Ehlers, Cybernetic Analysis For Stocks and Futures pg. 33
    params:
    @src: series, time-series input data
    @lag: integer, signal correction of n bar
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.cyber_cycle(df['ohlc4'],return_df=True)
    """
    src = src.dropna()
    n = len(src)
    alpha = .7
    alpha2 = 1/(lag+1)

    _df = pd.DataFrame({
        'close':src,
        'smooth':.00,
        'cycle':.00,
        'signal':.00
    }, index=src.index)

    for i in range(2, n):
        _df['smooth'][i] = (src[i]+2*src[i-1]+2*src[i-2]+src[i-3])/6
        _df['cycle'][i] = (1-.5*alpha)*(1-.5*alpha)*(_df['smooth'][i]-2*_df['smooth'][i-1]+_df['smooth'][i-2])+2*(1-alpha)*(_df['cycle'][i-2])
        _df['signal'][i] = alpha2*_df['cycle'][i]+(1-alpha2)*(_df['signal'][i-1])
    
    if return_df:
        return _df.iloc[lag:,:]
    else:
        return _df['signal'][lag:]

def ultimate_smoother(src, _length = 20, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers, 
    an enhanced smoother, as an evolution of his previously developed SuperSmoother,
    reference: https://traders.com/Documentation/FEEDbk_docs/2024/04/TradersTips.html
    params:
    @src: series, time-series input data
    @_length: integer, length of lookback period
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.ultimate_smoother(src=df['ohlc4'], _length=14, return_df=True)
    """
    src = src.dropna()
    n = len(src)
    
    if n < _length:
        raise ValueError('Periods cant be greater than data length')
    
    _df = pd.DataFrame({
        'close':src,
        'a1':0.00,
        'c2':0.00,
        'c3':0.00,
        'c1':0.00,
        'ultimate_smooth': 0.00
    }, index = src.index)
    
    _pi = 2*np.arcsin(1)
    _df['a1'] = math.exp(-1.414*_pi/_length)
    _df['c2'] = 2.0*_df['a1']*math.cos(1.414*_pi/_length)
    _df['c3'] = -_df['a1']*_df['a1']
    _df['c1'] = (1.0+_df['c2']-_df['c3'])/4.0
    _df['ultimate_smooth'] = _df['close']
    
    for i in range(4, n):
        _df['ultimate_smooth'][i] = (1.0-_df['c1'][i])*_df['close'][i]+(2.0*_df['c1'][i]-_df['c2'][i])*_df['close'][i-1]-(_df['c1'][i]+_df['c3'][i])*_df['close'][i-2]+_df['c2'][i]*_df['ultimate_smooth'][i-1]+_df['c3'][i]*_df['ultimate_smooth'][i-2]

    if return_df:
        return _df.iloc[_length:, :]
    else:
        return _df['ultimate_smooth'][_length:]

def instantaneous_trendline(src, alpha = .07, return_df = False):
    """
    technical analysis indicator:
    originated by John F. Ehlers, with aim to generate a responsive trend-following system,
    by utilize ITrend forced to be a finite impulse response (FIR)-smoothed version of price for the seven bars of calculation,
    and by adding trigger, the strategy is to open long position when trigger crossover the ITrend and vice versa
    reference: John F. Ehlers, Cybernetic Analysis for Stocks and Futures pg. 21
    params:
    @src: series, time-series input data
    @alpha: float, smoothing alpha
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.instantaneous_trendline(df['close'], return_df=True)
    """
    src = src.dropna()
    n = len(src)
    
    _df = pd.DataFrame({
        'close':src,
        'itrend': 0.00,
        'trigger': 0.00,
        'series': 0.00
    }, index = src.index)
    
    for i in range(7, n):
        _df['itrend'][i] = (alpha-alpha*alpha/4)*src[i]+.5*alpha*alpha*src[i-1]-(alpha-.75*alpha*alpha)*src[i-2]+2*(1-alpha)*_df['itrend'][i-1]-(1-alpha)*(1-alpha)*_df['itrend'][i-2]
        _df['trigger'][i] = 2*_df['itrend'][i]-_df['itrend'][i-2]
        
        if _df['itrend'][i] > _df['trigger'][i]:
            _df['series'][i] = _df['itrend'][i]
        else:
            _df['series'][i] = _df['trigger'][i]

    if return_df:
        return _df.iloc[(7*3):, :]
    else:
        return _df[['itrend', 'trigger', 'series']][(7*3):]

def entry_measure(src, threshold = .1):
    """
    risk management tool:
    return entry measumerement to relative high/low,
    assume in volatile market regime, 
    whether entry point is proper   
    params:
    @src: series, time-series input data
    @threshold: float, threshold to measure result
    example:
    >>> technical_indicator.entry_measure(df, threshold=.05)
    """
    
    n = len(src)
    
    _df = pd.DataFrame({
        'measure':.00,
        'distant':0
    }, index = src.index)
    
    for i in range(0, n):
        if (src['close'][i]/src['open'][i])-1 > 0:
            _df['measure'][i] = (src['close'][i]/src['low'][i])-1
        if (src['close'][i]/src['open'][i])-1 < 0:
            _df['measure'][i] = (src['close'][i]/src['high'][i])-1

    _df['distant'] = np.where(((_df['measure']>-threshold) & (_df['measure']<threshold)), False, True)
    
    return _df['distant']

def pma_atr_volatility_filter(src, periods = 10, threshold = .2):
    """
    risk management tool:
    return volatility regime analysis, 
    based on absolute value of difference between predict and trigger,
    from predictive moving average, 
    and observe if it above or below ATR * threshold,
    reference: personal development
    params:
    @src: series/df, time-series input data
    @periods: integer, lookback period
    @threshold: float, a multiplier for ATR
    example:
    >>> technical_indicator.pma_atr_volatility_filter(df)
    """
    src = src.dropna()
    n = len(src)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')

    src[['tr', 'atr']] = atr(src, periods = periods)
    src['pma_atr_volatile'] = True
    diff = abs(src['pma_predict']-src['pma_trigger'])
    src['pma_atr_volatile'] = np.where((diff>src['atr']*threshold), False, src['pma_atr_volatile'])

    src.dropna(inplace=True)
    return src

def rolling_slope(series, periods=10):
    """
    risk management tool:
    return trend strength detection,
    based on linear regression slope,
    reference: https://trendspider.com/learning-center/linear-regression-slope-a-comprehensive-guide-for-traders/
    params:
    @src: series/df, time-series input data
    @periods: integer, lookback period
    @threshold: float, a multiplier for ATR
    example:
    >>> technical_indicator.rolling_slope(df['ohlc4'], periods=14)
    """

    n = len(series)

    if n < periods:
        raise ValueError('Periods cant be greater than data length')
    
    slopes = []

    for i in range(n):
        if i < periods-1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-periods+1:i+1]
            x = np.arange(periods)
            x = x-x.mean()
            y = y-y.mean()
            slope = (x*y).sum()/(x**2).sum()
            slopes.append(slope)

    return pd.Series(slopes, index=series.index)

def heikin_ashi(src, return_df = False):
    """
    bar style:
    return a smoothed ohlc that filters out market noise,
    developed by Munehisa Homma,
    aim to easier spotting trend and reversal,
    reference: https://www.investopedia.com/terms/h/heikinashi.asp
    params:
    @src: series, time-series input data
    @return_df: boolean, whether to return include input dataframe or result only
    example:
    >>> technical_indicator.heikin_ashi(df,return_df=True)
    """
    src = src.dropna()
    n = len(src)
    src[['open_ha','high_ha','low_ha','close_ha']] = .00
    
    
    src['close_ha'] = (src['open']+src['high']+src['low']+src['close'])/4
    
    for i in range(0, n):
        if i==0:
            src['open_ha'][i]=((src['open'][i]+src['close_ha'][i])/2)
        else:
            src['open_ha'][i]=((src['open_ha'][i-1]+src['close_ha'][i-1])/2)
    
    src['high_ha'] = src[['open_ha','close_ha','high']].max(axis=1)
    src['low_ha'] = src[['open_ha','close_ha','low']].min(axis=1)
    
    if return_df:
        return src
    else:
        src = src[['open_ha','high_ha','low_ha','close_ha']]
        src.columns = ['open','high','low','close']
        return src



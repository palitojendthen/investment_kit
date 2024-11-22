<h2> How to use - example: </h2>

__load library__

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import yfinance as yf
import sys
path = r'path-to-investment_kit'
sys.path.insert(0, path)
import technical_indicator as indicator
import portfolio_management as portfolio
```

__retrieve data__

```python
start_date = '2021-1-1'; end_date = '2022-12-31'
df = yf.download(tickers = 'JPM', threads = True, start = start_date, end = end_date)
```

<br>

## compute simple decycler

```python
length = 89
df2 = indicator.simple_decycler(df['Close'], hp_period = length, return_df = True)
```

__visualize price to simple decycler indicator__
```python
df2['buy'] = np.where(df2['decycler'].pct_change() > 0, df2['decycler'], np.nan)
df2['sell'] = np.where((df2['decycler'].pct_change() < 0) | (df2['decycler'].pct_change() == 0), df2['decycler'], np.nan)
```

```python
ax = pd.Series(df['Close'][length:]).plot(color = 'black', figsize = (20,12))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('Dummy Stock Price Data vs Simple Decycler Indicator')
df2['buy'].plot.line(color = 'green')
df2['sell'].plot.line(color = 'red')
df2['hysteresis_up'].plot(color = 'orange')
df2['hysteresis_down'].plot(color = 'orange')
```

<img src="https://i.postimg.cc/FKVdYDW5/1.png" width=100% height=100%>

<br>

## compute predictive moving average
```python
df2 = indicator.predictive_moving_average(df['Close'], return_df = True)
```

__visualize price to predictive moving average__
```python
ax = df2['price'][-150:].plot(color = 'black', figsize = (20,12))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('Dummy Stock Price Data vs Predictive Moving Average Indicator')
df2['predict'][-150:].plot.line(color = 'green')
df2['trigger'][-150:].plot(color = 'red')
```

<img src="https://i.postimg.cc/cHjDrRLW/2.png" width=100% height=100%>

<br>

## compute even better sinewave
```python
df2 = indicator.even_better_sinewave(df['Close'], return_df = True)
```

__visualize price to even better sinwave__
```python
df2['buy'] = np.where((df2['wave'].shift(1) < df2['wave'].shift(0)) | (df2['wave'] > .8), df2['wave'], np.nan)
df2['sell'] = np.where((df2['wave'].shift(1) > df2['wave'].shift(0)) | (df2['wave'] < -.8), df2['wave'], np.nan)
df2
```

```python
fig = pt.figure(figsize = (20,12))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 

ax0 = pt.subplot(gs[0])
ax0.plot(df2.index[-200:], df2['price'][-200:], color = 'black')
ax1 = pt.subplot(gs[1])
ax1.plot(df2.index[-200:], df2['buy'][-200:], color = 'green')
ax1.plot(df2.index[-200:], df2['sell'][-200:], color = 'red')

ax0.yaxis.set_label_position("right")
ax0.yaxis.tick_right()
ax0.set_title('Dummy Stock Price Data vs Even Better Sinewave')
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()

fig.tight_layout()
```

<img src="https://i.postimg.cc/G3MHZXS9/3.png" width=100% height=100%>

<br>

## compute adaptive moving average
```python
start_date = '2021-1-1'; end_date = '2022-12-31'
df = yf.download(tickers = 'BTC-USD', threads = True, start = start_date, end = end_date)
```

```python
df2 = indicator.kama(df['Close'], return_df = True)
```

__visualize price to kaufman adaptive moving average indicator__
```python
df2['buy'] = np.where(df2['kama'].pct_change() > 0, df2['kama'], np.nan)
df2['sell'] = np.where((df2['kama'].pct_change() < 0) | (df2['kama'].pct_change() == 0), df2['kama'], np.nan)
```
```python
ax = pd.Series(df2['price']).plot(color = 'black', figsize = (20,12))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('BTCUSD vs KAMA Indicator')
df2['buy'].plot.line(color = 'green')
df2['sell'].plot.line(color = 'red')
```

<img src="https://i.postimg.cc/VkBzphtK/4.png" width=100% height=100%>

<h2> Description: </h2>

Collections of both Technical Analysis Indicator and Portfolio Management/Optimization writen in Python.


<h2> How to use - example: </h2>

__load library__

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import sys
path = r'path-to-investment_kit'
sys.path.insert(0, path)
import technical_indicator as indicator
import portfolio_management as portfolio
```

__retrieve data__

```python
df = pd.read_excel(open('dummy_data.xlsx', 'rb'), sheet_name = 'Sheet1', engine = 'openpyxl')
```

<br>

## compute simple decycler

```python
length = 89
df2 = indicator.simple_decycler(df['price'], hp_period = length, return_df = True)
```

__visualize price to simple decycler indicator__
```python
df2['buy'] = np.where(df2['decycler'].pct_change() > 0, df2['decycler'], np.nan)
df2['sell'] = np.where((df2['decycler'].pct_change() < 0) | (df2['decycler'].pct_change() == 0), df2['decycler'], np.nan)
```

```python
ax = pd.Series(df['price'][89:].values).plot(color = 'black', figsize = (15,8))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('Dummy Stock Price Data vs Simple Decycler Indicator')
df2['buy'].plot.line(color = 'green')
df2['sell'].plot.line(color = 'red')
df2['hysteresis_up'].plot(color = 'orange')
df2['hysteresis_down'].plot(color = 'orange')
```


<br>

## compute predictive moving average
```python
df2 = indicator.predictive_moving_average(df['price'], return_df = True)
```

__visualize price to predictive moving average__
```python
ax = df2['price'][14:].plot(color = 'black', figsize = (15,8))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('Dummy Stock Price Data vs Predictive Moving Average')
df2['predict'][14:].plot.line(color = 'green')
df2['trigger'][14:].plot(color = 'red')
```

<img src="https://i.postimg.cc/pTkzXmbV/Screenshot-2022-12-30-200658.png" width=100% height=100%>

<br>

## compute even better sinewave
```python
df = pd.read_excel(open('dummy_data2.xlsx', 'rb'), sheet_name = 'Sheet1', engine = 'openpyxl')
```

```python
df2 = investment.even_better_sinewave(df['close'], return_df = True)
```

__visualize price to even better sinwave__
```python
df2['buy'] = np.where((df2['wave'].shift(1) < df2['wave'].shift(0)) | (df2['wave'] > .8), df2['wave'], np.nan)
df2['sell'] = np.where((df2['wave'].shift(1) > df2['wave'].shift(0)) | (df2['wave'] < -.8), df2['wave'], np.nan)
df2
```

```python
fig = pt.figure(figsize = (12,8))
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

<img src="https://i.postimg.cc/8CHgYRvD/Screenshot-2023-01-01-170121.png" width=100% height=100%>

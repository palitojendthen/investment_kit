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
df2['decycler'].plot.line(color = 'blue')
df2['hysteresis_up'].plot(color = 'orange')
df2['hysteresis_down'].plot(color = 'orange')
```

<img src="https://i.postimg.cc/W4C59xHZ/Screenshot-2023-01-01-160558.png" width=100% height=100%>

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

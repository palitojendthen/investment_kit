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

__compute simple decycler__
```python
length = 89
df2 = investment.simple_decycler(df['price'].values, hp_period = length,return_df = True)
```

__visualize price to simple decycler indicator__
```python
ax = pd.Series(df['price'][89:].values).plot(color = 'black', figsize = (15,8))
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_title('Dummy Stock Price Data vs Simple Decycler Indicator')
df2['decycler'].plot.line(color = 'blue')
df2['hysteresis_up'].plot(color = 'orange')
df2['hysteresis_down'].plot(color = 'orange')
```

<img src="https://i.postimg.cc/qqNsn3tQ/Screenshot-2022-12-29-180250.png" width=100% height=100%>

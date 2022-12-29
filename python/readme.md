<h2> Description: </h2>

Collections of both Technical Analysis Indicator and Portfolio Management/Optimization writen in Python.


<h2> How to use: </h2>

__load library__

```python
import pandas as pd
import numpy as np
import matplotlib as pt
import sys
path = r'\path-to-investment_kit'
sys.path.insert(0, path)
import technical_indicator as indicator
import portfolio_management as portfol
```

__retrieve data__

```python
df = pd.read_excel(open('dummy_data.xlsx', 'rb'), sheet_name = 'Sheet1', engine = 'openpyxl')
```

__compute simple decycler__
```python
length = 89
indicator.simple_decycler(df['price'].values, hp_period = length, return_df = True)
```

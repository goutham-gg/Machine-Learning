import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

df = df[['Adj. Open','HL_PCT','PCT_change','Adj. Volume']]
forecast_col = 'adj.close'
df.fillna(-9999, inplace=True)

forecast_out=int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
x = x[:-forecast_out]

print(len(x), len(y))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("./data/ETT/ETTh1.csv").interpolate()
df["date"] = pd.to_datetime(df["date"])
df = df[['date',
             'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL','LULL',
             'OT']]
# df.set_index("date")


sd_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL','LULL']
for sd in sd_list:
    s = pd.Series(df[sd].values, index=df["date"])
    decompose = seasonal_decompose(s, two_sided=False) #period=12 #DatetimeIndex.inferred_freq -> freq_to_period #two_sided=False
    df[f"{sd}_seasonal"] = decompose.seasonal.values 

df.to_csv("./data/ETT/ETTh1_seasonal.csv")
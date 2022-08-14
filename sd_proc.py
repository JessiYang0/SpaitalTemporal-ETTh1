from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    df = pd.read_csv("./data/ETT/data_NA.csv").interpolate()
    df['date'] = df.apply(
        lambda x: f"{int(x['Year'])}-{int(x['Month'])}-{int(x['Day'])} {int(x['Hour'])}:00:00", axis=1)
    df = df[['date', 'Temperature', 'Pressure', 'Humidity', 'Wind_Speed', 'Rainfall', 'Season', 'Weekend', 'Load']]
    df["Date"] = pd.to_datetime(df["Date"])
    decompose = seasonal_decompose(df.values, period=3000)
    decompose.plot()
    print(decompose)


main()

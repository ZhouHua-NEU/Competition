import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_excel("ProblemB_time_series_60min_singleindex.xlsx")

load = df.loc[:, ["DE_load_actual_entsoe_transparency", "FR_load_actual_entsoe_transparency",
                  "GB_UKM_load_actual_entsoe_transparency", "IT_load_actual_entsoe_transparency"]]
load.columns = ["DE_load", "FR_load", "GB_load", "IT_load"]
solar = df.loc[:, ["DE_solar_generation_actual", "FR_solar_generation_actual",
                   "GB_UKM_solar_generation_actual", "IT_solar_generation_actual"]]
solar.columns = ["DE_solar", "FR_solar", "GB_solar", "IT_solar"]
wind = df.loc[:, ["DE_wind_generation_actual", "FR_wind_onshore_generation_actual",
                  "GB_UKM_wind_generation_actual", "IT_wind_onshore_generation_actual"]]
wind.columns = ["DE_wind", "FR_wind", "GB_wind", "IT_wind"]

time = df.iloc[:, 1]
time.columns = "time"
data = pd.concat([time, load, solar, wind], axis=1)

data.isna().sum()
data = data.fillna(method="pad")
data.isna().sum()
data = data.fillna(method="bfill")
data.isna().sum()

def sum_data(df, i, n):
    data = df.iloc[:, i]
    ls = []
    j = 0
    while j+n-1 < len(data):
        ls.append(data[j: j+n].sum(axis=0))
        j = j+n
    return pd.Series(ls)

DE_load = sum_data(df, 2, 24*7)
FR_load = sum_data(df, 3, 24*7)
GB_load = sum_data(df, 4, 24*7)
IT_load = sum_data(df, 5, 24*7)
DE_solar = sum_data(df, 6, 24*7)
FR_solar = sum_data(df, 7, 24*7)
GB_solar = sum_data(df, 8, 24*7)
IT_solar = sum_data(df, 9, 24*7)
DE_wind = sum_data(df, 10, 24*7)
FR_wind = sum_data(df, 11, 24*7)
GB_wind = sum_data(df, 12, 24*7)
IT_wind = sum_data(df, 13, 24*7)

adfuller(DE_load)
adfuller(FR_load)
adfuller(GB_load)
adfuller(IT_load)
adfuller(DE_solar)
adfuller(FR_solar)
adfuller(GB_solar)
adfuller(IT_solar)
adfuller(DE_wind)
adfuller(FR_wind)
adfuller(GB_wind)
adfuller(IT_wind)
adfuller(GB_load.diff().dropna())
adfuller(FR_wind.diff().dropna())

def plot_cf(data):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(data, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(data, lags=20, ax=ax2)
    plt.show()

plot_cf(GB_wind)
plot_cf(IT_solar)
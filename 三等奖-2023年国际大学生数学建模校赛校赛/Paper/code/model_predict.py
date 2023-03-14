import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("four_country.csv")

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

def arima(data, d):
    pmax = int(len(data)/100)
    qmax = int(len(data)/100)
    bic_matrix = []
    for p in range(pmax + 1):
        temp = []
        for q in range(qmax+1):
            try:
                temp.append(ARIMA(data, order=(p, d, q)).fit().bic)
            except:
                temp.append(None)
            bic_matrix.append(temp)
    bic_matrix = pd.DataFrame(bic_matrix)
    bic_matrix = bic_matrix.astype(float)
    p, q = bic_matrix.stack().idxmin()
    model = ARIMA(data, order=(p, d, q)).fit()
    return model, pd.Series(p), pd.Series(q)

model_DE, DE_p, DE_q = arima(DE_load, 0)
model_FR, FR_p, FR_q = arima(FR_load, 0)
model_GB, GB_p, GB_q = arima(GB_load, 0)
model_IT, IT_p, IT_q = arima(IT_load, 0)
pa1 = pd.concat([DE_p, DE_q, FR_p, FR_q,
                 GB_p, GB_q, IT_p, IT_q], axis=1)

DE_pre = model_DE.predict(0, len(DE_load)+5)
FR_pre = model_FR.predict(0, len(FR_load)+5)
GB_pre = model_GB.predict(0, len(GB_load)+5)
IT_pre = model_IT.predict(0, len(IT_load)+5)
l1 = pd.concat([DE_pre[-5:], FR_pre[-5:], GB_pre[-5:], IT_pre[-5:]], axis=1)
l1.columns = ["DE_loadpre", "FR_loadpre", "GB_loadpre", "IT_loadpre"]

plt.figure(figsize=(20, 10), dpi=100)
plt.subplot(221)
plt.plot(range(len(DE_load)), DE_load, label="DE_load")
plt.plot(range(len(DE_pre)-5), DE_pre[:-5], label="DE_pre")
plt.plot(range(len(DE_pre)-5, len(DE_pre)), DE_pre[-5:], label="DE_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(222)
plt.plot(range(len(FR_load)), FR_load, label="FR_load")
plt.plot(range(len(FR_pre)-5), FR_pre[:-5], label="FR_pre")
plt.plot(range(len(FR_pre)-5, len(FR_pre)), FR_pre[-5:], label="FR_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(223)
plt.plot(range(len(GB_load)), GB_load, label="GB_load")
plt.plot(range(len(GB_pre)-5), GB_pre[:-5], label="GB_pre")
plt.plot(range(len(GB_pre)-5, len(GB_pre)), GB_pre[-5:], label="GB_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(224)
plt.plot(range(len(IT_load)), IT_load, label="IT_load")
plt.plot(range(len(IT_pre)-5), IT_pre[:-5], label="IT_pre")
plt.plot(range(len(IT_pre)-5, len(IT_pre)), IT_pre[-5:], label="IT_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()
plt.savefig("load.jpg")
plt.show()

model_DE, DE_p, DE_q = arima(DE_solar, 0)
model_FR, FR_p, FR_q = arima(FR_solar, 0)
model_GB, GB_p, GB_q = arima(GB_solar, 0)
model_IT, IT_p, IT_q = arima(IT_solar, 0)
pa2 = pd.concat([DE_p, DE_q, FR_p, FR_q,
                 GB_p, GB_q, IT_p, IT_q], axis=1)

DE_pre = model_DE.predict(0, len(DE_solar)+5)
FR_pre = model_FR.predict(0, len(FR_solar)+5)
GB_pre = model_GB.predict(0, len(GB_solar)+5)
IT_pre = model_IT.predict(0, len(IT_solar)+5)
l2 = pd.concat([DE_pre[-5:], FR_pre[-5:], GB_pre[-5:], IT_pre[-5:]], axis=1)
l2.columns = ["DE_solarpre", "FR_solarpre", "GB_solarpre", "IT_solarpre"]

plt.figure(figsize=(20, 10), dpi=100)
plt.subplot(221)
plt.plot(range(len(DE_solar)), DE_solar, label="DE_solar")
plt.plot(range(len(DE_pre)-5), DE_pre[:-5], label="DE_pre")
plt.plot(range(len(DE_pre)-5, len(DE_pre)), DE_pre[-5:], label="DE_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(222)
plt.plot(range(len(FR_solar)), FR_solar, label="FR_solar")
plt.plot(range(len(FR_pre)-5), FR_pre[:-5], label="FR_pre")
plt.plot(range(len(FR_pre)-5, len(FR_pre)), FR_pre[-5:], label="FR_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(223)
plt.plot(range(len(GB_solar)), GB_solar, label="GB_solar")
plt.plot(range(len(GB_pre)-5), GB_pre[:-5], label="GB_pre")
plt.plot(range(len(GB_pre)-5, len(GB_pre)), GB_pre[-5:], label="GB_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(224)
plt.plot(range(len(IT_solar)), IT_solar, label="IT_solar")
plt.plot(range(len(IT_pre)-5), IT_pre[:-5], label="IT_pre")
plt.plot(range(len(IT_pre)-5, len(IT_pre)), IT_pre[-5:], label="IT_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()
plt.savefig("solar.jpg")
plt.show()

model_DE, DE_p, DE_q = arima(DE_wind, 0)
model_FR, FR_p, FR_q = arima(FR_wind, 0)
model_GB, GB_p, GB_q = arima(GB_wind, 0)
model_IT, IT_p, IT_q = arima(IT_wind, 0)
pa3 = pd.concat([DE_p, DE_q, FR_p, FR_q,
                 GB_p, GB_q, IT_p, IT_q], axis=1)

DE_pre = model_DE.predict(0, len(DE_wind)+5)
FR_pre = model_FR.predict(0, len(FR_wind)+5)
GB_pre = model_GB.predict(0, len(GB_wind)+5)
IT_pre = model_IT.predict(0, len(IT_wind)+5)
l3 = pd.concat([DE_pre[-5:], FR_pre[-5:], GB_pre[-5:], IT_pre[-5:]], axis=1)
l3.columns = ["DE_windpre", "FR_windpre", "GB_windpre", "IT_windpre"]

plt.figure(figsize=(20, 10), dpi=100)
plt.subplot(221)
plt.plot(range(len(DE_wind)), DE_wind, label="DE_wind")
plt.plot(range(len(DE_pre)-5), DE_pre[:-5], label="DE_pre")
plt.plot(range(len(DE_pre)-5, len(DE_pre)), DE_pre[-5:], label="DE_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(222)
plt.plot(range(len(FR_wind)), FR_wind, label="FR_wind")
plt.plot(range(len(FR_pre)-5), FR_pre[:-5], label="FR_pre")
plt.plot(range(len(FR_pre)-5, len(FR_pre)), FR_pre[-5:], label="FR_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(223)
plt.plot(range(len(GB_wind)), GB_wind, label="GB_wind")
plt.plot(range(len(GB_pre)-5), GB_pre[:-5], label="GB_pre")
plt.plot(range(len(GB_pre)-5, len(GB_pre)), GB_pre[-5:], label="GB_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()

plt.subplot(224)
plt.plot(range(len(IT_wind)), IT_wind, label="IT_wind")
plt.plot(range(len(IT_pre)-5), IT_pre[:-5], label="IT_pre")
plt.plot(range(len(IT_pre)-5, len(IT_pre)), IT_pre[-5:], label="IT_forecast")
plt.xticks([0, 52, 104, 156, 209, 260], [2015, 2016, 2017, 2018, 2019, 2020])
plt.xlabel("time")
plt.ylabel("load")
plt.legend()
plt.savefig("wind.jpg")
plt.show()

l1.index = range(len(l1))
l2.index = range(len(l2))
l3.index = range(len(l3))
table = pd.concat([l1, l2, l3], axis=1)
table.index = range(len(table))
table = table.astype(int)
print(table.to_latex())

DE_solar_ra = table.apply(lambda x: x["DE_solarpre"]/x["DE_loadpre"], axis=1)
DE_wind_ra = table.apply(lambda x: x["DE_windpre"]/x["DE_loadpre"], axis=1)
DE_sw_ratio = table.apply(lambda x: x["DE_solarpre"]/x["DE_windpre"], axis=1)
FR_solar_ra = table.apply(lambda x: x["FR_solarpre"]/x["FR_loadpre"], axis=1)
FR_wind_ra = table.apply(lambda x: x["FR_windpre"]/x["FR_loadpre"], axis=1)
FR_sw_ratio = table.apply(lambda x: x["FR_solarpre"]/x["FR_windpre"], axis=1)
GB_solar_ra = table.apply(lambda x: x["GB_solarpre"]/x["GB_loadpre"], axis=1)
GB_wind_ra = table.apply(lambda x: x["GB_windpre"]/x["GB_loadpre"], axis=1)
GB_sw_ratio = table.apply(lambda x: x["GB_solarpre"]/x["GB_windpre"], axis=1)
IT_solar_ra = table.apply(lambda x: x["IT_solarpre"]/x["IT_loadpre"], axis=1)
IT_wind_ra = table.apply(lambda x: x["IT_windpre"]/x["IT_loadpre"], axis=1)
IT_sw_ratio = table.apply(lambda x: x["IT_solarpre"]/x["IT_windpre"], axis=1)
ratio_table = pd.concat([DE_solar_ra, DE_wind_ra, DE_sw_ratio,
                         FR_solar_ra, FR_wind_ra, FR_sw_ratio,
                         GB_solar_ra, GB_wind_ra, GB_sw_ratio,
                         IT_solar_ra, IT_wind_ra, IT_sw_ratio], axis=1)
ratio_table.columns = ["DE_solar_ra", "DE_wind_ra", "DE_sw_ratio",
                       "FR_solar_ra", "FR_wind_ra", "FR_sw_ratio",
                       "GB_solar_ra", "GB_wind_ra", "GB_sw_ratio",
                       "IT_solar_ra", "IT_wind_ra", "IT_sw_ratio"]

ratio_table = round(ratio_table, 2)
print(ratio_table.to_latex())


pas = pd.concat([pa1, pa2, pa3], axis=0)
pas.columns = ["DE_p", "DE_q", "FR_p", "FR_q",
               "GB_p", "GB_q", "IT_p", "IT_q"]
pas.index = ["load", "solar", "wind"]
print(pas.to_latex())
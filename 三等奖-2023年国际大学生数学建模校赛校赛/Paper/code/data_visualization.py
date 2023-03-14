import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("four.csv")

data=data.fillna(method = "pad")
data=data.fillna(method = "bfill")
data.isna().sum()

data.tail(5)

def sum_data(df, i, n):
    data = df.iloc[:, i]
    ls = []
    j = 0
    while j+n-1 < len(data):
        ls.append(data[j: j+n].sum(axis=0))
        j = j+n
    return pd.Series(ls)

index=pd.date_range(start='2015-01-01', end = '2020-9-30',freq = "W",name = "time")
data_day=pd.DataFrame([sum_data(data,i,7*24) for i in range(1,13) ]).T
data_day.index = index
data_day.columns=['DE_load_actual_entsoe_transparency','GB_UKM_load_actual_entsoe_transparency','FR_load_actual_entsoe_transparency','IT_load_actual_entsoe_transparency',
                  'DE_wind_generation_actual','GB_UKM_wind_generation_actual','FR_wind_onshore_generation_actual','IT_wind_onshore_generation_actual',
                  'DE_solar_generation_actual','GB_UKM_solar_generation_actual','FR_solar_generation_actual','IT_solar_generation_actual']

a=data_day.describe().round(0)
a.columns=['DE_load','GB_UKM_load','FR_load','IT_load',
                  'DE_wind','GB_UKM','FR_wind','IT_wind',
                  'DE_solar','GB_UKM_solar','FR_solar','IT_solar']
a.T
a.T.to_latex()

plt.figure(figsize=(15,9),dpi=120)
ax3=plt.subplot(221)
plt.boxplot(data_day[['DE_load_actual_entsoe_transparency','DE_wind_generation_actual','DE_solar_generation_actual']])
plt.grid(linestyle="--", alpha=0.3)
ax3.set_xticklabels(['DE_load','DE_wind','DE_solar'], fontsize=8)
ax1=plt.subplot(222)
plt.boxplot(data_day[['GB_UKM_load_actual_entsoe_transparency','GB_UKM_wind_generation_actual','GB_UKM_solar_generation_actual']])
plt.grid(linestyle="--", alpha=0.3)
ax1.set_xticklabels(['GB_UKM_load','GB_UKM_wind','GB_UKM_solar'], fontsize=8)
ax2=plt.subplot(223)
plt.boxplot(data_day[['FR_load_actual_entsoe_transparency','FR_wind_onshore_generation_actual','FR_solar_generation_actual']])
plt.grid(linestyle="--", alpha=0.3)
ax2.set_xticklabels(['FR_load','FR_wind','FR_solar'], fontsize=8)
ax4=plt.subplot(224) 
plt.boxplot(data_day[['IT_load_actual_entsoe_transparency','IT_wind_onshore_generation_actual','IT_solar_generation_actual']])
plt.grid(linestyle="--", alpha=0.3)
ax4.set_xticklabels(['IT_load','IT_wind','IT_solar'], fontsize=8)          
plt.savefig(r".\7.png")
plt.show()

import seaborn as sns
sns.set_style("white")
j=1
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6), dpi=120)
plt.xlabel('time(week)')
plt.ylabel('load(MW)')
data_day['DE_load_actual_entsoe_transparency'].plot(color=sns.hls_palette(50, l=.7, s=.9)[j])
j+=5
data_day['GB_UKM_load_actual_entsoe_transparency'].plot(color=sns.hls_palette(50, l=.7, s=.9)[j])
j+=5
data_day['FR_load_actual_entsoe_transparency'].plot(color=sns.hls_palette(50, l=.7, s=.9)[j])
j+=5
data_day['IT_load_actual_entsoe_transparency'].plot(color=sns.hls_palette(50, l=.7, s=.9)[j])
j+=5          
plt.legend()
plt.savefig(r".\images\1.png")

sns.set_style("ticks")
plt.figure(figsize=(10, 6), dpi=120)
plt.xlabel('time(week)')
plt.ylabel('wind_generation(MW)')
j=3
for i in['DE_wind_generation_actual','GB_UKM_wind_generation_actual','FR_wind_onshore_generation_actual','IT_wind_onshore_generation_actual']:
    data_day[i].plot(color=sns.hls_palette(200, l=.7, s=.9)[j])
    j+=20
plt.legend()
plt.savefig(r".\images\2.png")

sns.set_style("white")
plt.figure(figsize=(10, 6), dpi=120)
plt.xlabel('time(week)')
plt.ylabel('wind_generation(MW)')
j=2
for i in['DE_solar_generation_actual','GB_UKM_solar_generation_actual','FR_solar_generation_actual','IT_solar_generation_actual']:
    data_day[i].plot(color=sns.hls_palette(100, l=.7, s=.9)[j])
    j+=12 
plt.legend()
plt.savefig(r".\images\3.png")

plt.figure(figsize=(15,9),dpi=120)
j=0
plt.subplot(221)
for i in['DE_load_actual_entsoe_transparency','DE_wind_generation_actual','DE_solar_generation_actual']:
    data_day[i].plot(color=sns.color_palette("hls",12)[j])
    j+=1
plt.legend()
plt.subplot(222)
for i in['GB_UKM_load_actual_entsoe_transparency','GB_UKM_wind_generation_actual','GB_UKM_solar_generation_actual']:
    data_day[i].plot(color=sns.color_palette("hls",12)[j])
    j+=1
plt.legend()
plt.subplot(223)
for i in['FR_load_actual_entsoe_transparency','FR_wind_onshore_generation_actual','FR_solar_generation_actual']:
    data_day[i].plot(color=sns.color_palette("hls",12)[j])
    j+=1
plt.legend()
plt.subplot(224) 
for i in['IT_load_actual_entsoe_transparency','IT_wind_onshore_generation_actual','IT_solar_generation_actual']:
    data_day[i].plot()
plt.legend()
plt.savefig(r".\images\4.png")
plt.show()

plt.figure(figsize=(15,9),dpi=120)
plt.subplot(221)
(data_day['DE_wind_generation_actual']/data_day['DE_load_actual_entsoe_transparency']).plot(label='DE_wind_persent',color=sns.color_palette("Paired",8)[0])
(data_day['DE_solar_generation_actual']/data_day['DE_load_actual_entsoe_transparency']).plot(label='DE_solar_persent',color=sns.color_palette("Paired",8)[1])
plt.legend()
plt.subplot(222)
(data_day['GB_UKM_wind_generation_actual']/data_day['GB_UKM_load_actual_entsoe_transparency']).plot(label='GB_UKM_wind_persent',color=sns.color_palette("Paired",8)[2])
(data_day['GB_UKM_solar_generation_actual']/data_day['GB_UKM_load_actual_entsoe_transparency']).plot(label='GB_UKM_solar_persent',color=sns.color_palette("Paired",8)[3])
plt.legend()
plt.subplot(223)
(data_day['FR_wind_onshore_generation_actual']/data_day['FR_load_actual_entsoe_transparency']).plot(label='FR_wind_persent',color=sns.color_palette("Paired",8)[4])
(data_day['FR_solar_generation_actual']/data_day['FR_load_actual_entsoe_transparency']).plot(label='FR_solar_persent',color=sns.color_palette("Paired",8)[5])
plt.legend()
plt.subplot(224)
(data_day['IT_wind_onshore_generation_actual']/data_day['IT_load_actual_entsoe_transparency']).plot(label='IT_wind_persent',color=sns.color_palette("Paired",8)[6])
(data_day['IT_solar_generation_actual']/data_day['IT_load_actual_entsoe_transparency']).plot(label='IT_solar_persent',color=sns.color_palette("Paired",8)[7])
plt.legend()
plt.savefig(r".\images\5.png")

plt.figure(figsize=(15,9),dpi=120)
plt.subplot(221)
((data_day['DE_solar_generation_actual']+data_day['DE_wind_generation_actual'])/data_day['DE_load_actual_entsoe_transparency']).plot(label='DE_green_energy_pensent',color=sns.cubehelix_palette(4, start=.5, rot=-.75)[1])
plt.legend()
plt.subplot(222)
((data_day['GB_UKM_solar_generation_actual']+data_day['GB_UKM_wind_generation_actual'])/data_day['GB_UKM_load_actual_entsoe_transparency']).plot(label='GB_green_energy_pensent',color=sns.cubehelix_palette(4, start=.5, rot=-.75)[2])
plt.legend()
plt.subplot(223)
((data_day['FR_solar_generation_actual']+data_day['FR_wind_onshore_generation_actual'])/data_day['FR_load_actual_entsoe_transparency']).plot(label='FR_green_energy_pensent',color=sns.cubehelix_palette(4, start=.5, rot=-.75)[3])
plt.legend()
plt.subplot(224)
((data_day['IT_solar_generation_actual']+data_day['IT_wind_onshore_generation_actual'])/data_day['IT_load_actual_entsoe_transparency']).plot(label='IT_green_energy_pensent',color=sns.cubehelix_palette(4, start=.5, rot=-.75)[0])
plt.legend()
plt.savefig(r".\images\6.png")







































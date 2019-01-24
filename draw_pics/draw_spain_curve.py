# -*- coding: utf-8 -*-
import build_spain_HPI_index_dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import datetime
data = build_spain_HPI_index_dataset.index_temp
date = build_spain_HPI_index_dataset.date_tmp

dates = [datetime.datetime.strptime(d, '%Y-%m').date() for d in date]
print(dates)

plt.figure(figsize=(18, 9))
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mpl.dates.YearLocator())
plt.xticks(pd.date_range(str(dates[0]), str(dates[-1]), freq='6M'))
plt.xticks(rotation=30)
plt.plot(dates, data)
date1 = dates[:100]
data1 = data[:100]
date2 = dates[100:]
data2 = data[100:]
plt.scatter(date1, data1, s=20, c='r',marker="x")
plt.scatter(date2, data2, s=20, c='r',marker="o")
ax1.set_title("Spain HPI" + date[0])

plt.show()

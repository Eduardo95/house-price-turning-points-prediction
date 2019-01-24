# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import datetime
case_s_data = pd.read_pickle('data_directory\case_shiller_data_and_turning_points.pkl')
original_cs_index = case_s_data['case_shiller_original'][11:]  # 应该是11
date = case_s_data.index[11:]
tp = case_s_data['tp_class_one_month'][11:]
data = original_cs_index
dates = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in date]
print(dates)

date1 = list()
data1 = list()
date2 = list()
data2 = list()
tp_date = list()
tp_data = list()
u_tp_date = list()
u_tp_data = list()
for i in range(328, 358, 1):  # 应该是328和358
    if tp[i] == 2 or tp[i] == 0:
        tp_date.append(dates[i])
        tp_data.append(data[i])
    elif tp[i] == 1:
        u_tp_date.append(dates[i])
        u_tp_data.append(data[i])
print(len(tp_date))
print(len(u_tp_date))

# date1.append(tp_date[6])
# data1.append(tp_data[6])
date1.append(u_tp_date[14])
data1.append(u_tp_data[14])
date1.append(u_tp_date[15])
data1.append(u_tp_data[15])
date1.append(u_tp_date[20])
data1.append(u_tp_data[20])
# for i in range(len(tp_date)):
#     if i != 6:
#         date2.append(tp_date[i])
#         data2.append(tp_data[i])
for i in range(len(u_tp_date)):
    if i != 14 and i != 15 and i != 20:
        date2.append(u_tp_date[i])
        data2.append(u_tp_data[i])

print(len(date1))
print(len(data1))
print(len(date2))
print(len(data2))
plt.figure(figsize=(18, 9))
gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mpl.dates.YearLocator())
plt.xticks(pd.date_range(str(dates[0]), str(dates[-1]), freq='6M'))
plt.xticks(rotation=30)
plt.plot(dates, data)
# plt.scatter(date1, data1, s=40, c='r', marker="x")
# plt.scatter(date2, data2, s=40, c='k', marker="o")
ax1.set_title("American Case-Shiller Index")
print(dates)
for i in range(len(data)):
    print(data.index[i])
    print(data[i])
print(tp[328:])
plt.show()

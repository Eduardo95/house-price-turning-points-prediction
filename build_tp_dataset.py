# -*- coding: utf-8 -*-
import pandas as pd
import case_shiller_data
import numpy as np
'''
这个文件是用来建立数据集，包括拐点数据和原始数据等
'''
ts = pd.DataFrame()
ts.loc[:, 'case_shiller_original'] = [i for i in case_shiller_data.data]
ts_test = ts['case_shiller_original']
ts_test_log = np.log(ts_test)
# 取对数以后再滑动平均，降低平稳性
moving_avg = pd.rolling_mean(ts_test_log, 12)
# 取对数以后滑动平均，再取对数值和滑动平均之间的偏差
ts_log_moving_avg_diff = ts_test_log - moving_avg
ts.loc[:, 'log_moving_avg_diff'] = [i for i in ts_log_moving_avg_diff]
ts.loc[:, 'tp_class_one_month'] = [1 for i in range(ts.iloc[:, 0].size)]
ts.index = [value for value in case_shiller_data.date]
tp_class = list()
tp_class.append(1)
for i in range(ts.iloc[:, 0].size - 2):
    if ts['case_shiller_original'][i + 1] > ts['case_shiller_original'][i] and \
            ts['case_shiller_original'][i + 1] > ts['case_shiller_original'][i + 2]:
        tp_class.append(2)
    elif ts['case_shiller_original'][i + 1] < ts['case_shiller_original'][i] and \
            ts['case_shiller_original'][i + 1] < ts['case_shiller_original'][i + 2]:
        tp_class.append(0)
    else:
        tp_class.append(1)
tp_class.append(1)
ts.loc[:, 'tp_class_one_month'] = [i for i in tp_class]

tp_class_in_3_month = list()
tp_class_in_3_month.append(1)
for i in range(ts.iloc[:, 0].size - 1 - 3):
    if ts['case_shiller_original'][i + 2] > ts['case_shiller_original'][i + 1] and \
            ts['case_shiller_original'][i + 3] > ts['case_shiller_original'][i + 4]:
        tp_class_in_3_month.append(2)
    elif ts['case_shiller_original'][i + 2] < ts['case_shiller_original'][i + 1] and \
            ts['case_shiller_original'][i + 3] < ts['case_shiller_original'][i + 4]:
        tp_class_in_3_month.append(0)
    else:
        tp_class_in_3_month.append(1)
tp_class_in_3_month.append(1)
tp_class_in_3_month.append(1)
tp_class_in_3_month.append(1)
ts.loc[:, 'tp_class_in_3_month'] = [i for i in tp_class_in_3_month]

tp_class_selected = [1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1,
                     1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                     1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                     1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                     1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 2,
                     1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1,
                     1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                     1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                     1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                     1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                     0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2,
                     1, 0, 1, 1, 1, 1, 1, 2, 0, 1]
ts.loc[:, 'tp_class_selected'] = [i for i in tp_class_selected]

# 经过min-max归一化的数据
scaled = list()
scaled.append(0.0)
for i in range(ts.iloc[:, 0].size-1):
    max_v = max(ts_test[0:i + 1 + 1])
    min_v = min(ts_test[0:i + 1 + 1])
    scaled.append((ts_test[i + 1] - min_v) / (max_v - min_v))
ts.loc[:, 'min-max'] = [i for i in scaled]

# =======================================================
# 经过处理后的数据格式如下：
#        case_shiller_original  log_moving_avg_diff  tp_class_one_month  tp_class_in_3_month  tp_class_selected  min-max
# timestamp1
# timestamp2
# timestamp3
# ......         ......               ......                          ......
# ========================================================

print(ts)
ts.to_pickle('data_directory\case_shiller_data_and_turning_points.pkl')

# -*- coding: utf-8 -*-
import pandas as pd
'''
这个文件是用来建立和谷歌搜索指数相关的数据集
'''
date = ['2004-01', '2004-02', '2004-03', '2004-04',
        '2004-05', '2004-06', '2004-07', '2004-08',
        '2004-09', '2004-10', '2004-11', '2004-12',
        '2005-01', '2005-02', '2005-03', '2005-04',
        '2005-05', '2005-06', '2005-07', '2005-08',
        '2005-09', '2005-10', '2005-11', '2005-12',
        '2006-01', '2006-02', '2006-03', '2006-04',
        '2006-05', '2006-06', '2006-07', '2006-08',
        '2006-09', '2006-10', '2006-11', '2006-12',
        '2007-01', '2007-02', '2007-03', '2007-04',
        '2007-05', '2007-06', '2007-07', '2007-08',
        '2007-09', '2007-10', '2007-11', '2007-12',
        '2008-01', '2008-02', '2008-03', '2008-04',
        '2008-05', '2008-06', '2008-07', '2008-08',
        '2008-09', '2008-10', '2008-11', '2008-12',
        '2009-01', '2009-02', '2009-03', '2009-04',
        '2009-05', '2009-06', '2009-07', '2009-08',
        '2009-09', '2009-10', '2009-11', '2009-12',
        '2010-01', '2010-02', '2010-03', '2010-04',
        '2010-05', '2010-06', '2010-07', '2010-08',
        '2010-09', '2010-10', '2010-11', '2010-12',
        '2011-01', '2011-02', '2011-03', '2011-04',
        '2011-05', '2011-06', '2011-07', '2011-08',
        '2011-09', '2011-10', '2011-11', '2011-12',
        '2012-01', '2012-02', '2012-03', '2012-04',
        '2012-05', '2012-06', '2012-07', '2012-08',
        '2012-09', '2012-10', '2012-11', '2012-12',
        '2013-01', '2013-02', '2013-03', '2013-04',
        '2013-05', '2013-06', '2013-07', '2013-08',
        '2013-09', '2013-10', '2013-11', '2013-12',
        '2014-01', '2014-02', '2014-03', '2014-04',
        '2014-05', '2014-06', '2014-07', '2014-08',
        '2014-09', '2014-10', '2014-11', '2014-12',
        '2015-01', '2015-02', '2015-03', '2015-04',
        '2015-05', '2015-06', '2015-07', '2015-08',
        '2015-09', '2015-10', '2015-11', '2015-12',
        '2016-01', '2016-02', '2016-03', '2016-04',
        '2016-05', '2016-06', '2016-07', '2016-08',
        '2016-09', '2016-10', '2016-11', '2016-12',
        '2017-01', '2017-02', '2017-03', '2017-04',
        '2017-05', '2017-06', '2017-07', '2017-08',
        '2017-09', '2017-10', '2017-11', '2017-12',
        '2018-01', '2018-02', '2018-03', '2018-04',
        '2018-05', '2018-06', '2018-07']
home_price = [82, 78, 74, 89, 78, 84, 89, 88, 78, 79, 80, 75, 79, 77, 73, 91,
              84, 96, 90, 89, 79, 76, 73, 67, 70, 77, 77, 80, 76, 73, 87, 73,
              68, 74, 60, 67, 70, 67, 72, 75, 69, 69, 80, 76, 71, 70, 77, 70,
              72, 81, 78, 82, 76, 81, 84, 78, 73, 78, 72, 70, 72, 78, 77, 73,
              78, 69, 70, 68, 63, 73, 72, 61, 64, 63, 66, 67, 71, 59, 55, 65,
              55, 52, 59, 55, 67, 63, 68, 68, 74, 66, 63, 61, 61, 59, 57, 57,
              58, 65, 65, 66, 62, 64, 62, 64, 63, 58, 68, 58, 65, 62, 69, 70,
              76, 73, 70, 71, 62, 65, 65, 66, 67, 65, 69, 72, 75, 74, 72, 73,
              70, 64, 73, 70, 75, 68, 75, 82, 78, 77, 80, 82, 75, 70, 74, 73,
              73, 76, 82, 85, 83, 81, 81, 74, 73, 72, 73, 75, 79, 80, 88, 87,
              93, 89, 93, 87, 92, 81, 92, 96, 91, 92, 90, 94, 98, 95, 100]
house_price = [38, 32, 44, 43, 45, 35, 45, 39, 39, 40, 46, 37, 40, 47, 43, 42,
               49, 49, 51, 44, 41, 37, 44, 36, 42, 42, 48, 44, 47, 45, 41, 41,
               35, 43, 36, 31, 43, 39, 39, 41, 43, 41, 39, 41, 41, 39, 43, 39,
               42, 42, 44, 40, 41, 45, 44, 38, 41, 46, 49, 45, 41, 41, 42, 43,
               40, 39, 36, 41, 33, 42, 40, 38, 35, 35, 39, 40, 36, 37, 35, 36,
               33, 39, 38, 35, 39, 35, 36, 41, 41, 38, 40, 38, 36, 41, 41, 37,
               36, 40, 37, 38, 40, 39, 41, 42, 42, 37, 40, 36, 37, 38, 40, 42,
               45, 42, 44, 41, 40, 48, 42, 38, 41, 41, 39, 47, 46, 47, 46, 47,
               39, 44, 43, 44, 42, 41, 47, 49, 51, 51, 49, 54, 49, 48, 48, 45,
               49, 51, 54, 58, 55, 55, 56, 49, 50, 52, 48, 52, 61, 56, 59, 58,
               59, 62, 62, 54, 58, 57, 57, 61, 62, 57, 59, 64, 61, 59, 57]
ts = pd.DataFrame()
ts.loc[:, 'home_price'] = [i for i in home_price]
ts.loc[:, 'house_price'] = [i for i in house_price]
ts.index = [value for value in date]

print(ts)
ts.to_pickle('data_directory\google_trend_data.pkl')
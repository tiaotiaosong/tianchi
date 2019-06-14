import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime


users = []
abnormal = []
wifis = []
shops = []
data = pd.read_csv('m_690data.csv')

shopdata = pd.read_csv('ccf_first_round_shop_info.csv',header=0)

longmin = np.min(shopdata[shopdata.mall_id=='m_7800'].longitude)
longmax = np.max(shopdata[shopdata.mall_id=='m_7800'].longitude)
longmean = np.mean(shopdata[shopdata.mall_id=='m_7800'].longitude)
latimin = np.min(shopdata[shopdata.mall_id=='m_7800'].latitude)
latimax = np.max(shopdata[shopdata.mall_id=='m_7800'].latitude)
latimean = np.mean(shopdata[shopdata.mall_id=='m_7800'].latitude)
print shopdata.ix[:,['longitude','latitude']]
print shopdata['longitude']
# x1_min, x2_min = np.min(shopdata[:,['longitude','latitude']],axis = 0)
# x1_max, x2_max = np.max(shopdata[:,['longitude','latitude']],axis = 0)
# print x1_min , x1_max
# print x2_min , x2_max
# plt.scatter(data['d'], data['a'], c=None, s=10, cmap=None, edgecolors='none')
# plt.xlim((x1_min+1, x1_max+1))
# plt.ylim((x2_min+1, x2_max+1))
# plt.grid(True)
# plt.show()
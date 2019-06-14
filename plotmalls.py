import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('m_690data.csv')
shopdata = pd.read_csv('ccf_first_round_shop_info.csv',header=0)
print data.shape
longmin = np.min(shopdata[shopdata.mall_id=='m_690'].longitude)
longmax = np.max(shopdata[shopdata.mall_id=='m_690'].longitude)
latimin = np.min(shopdata[shopdata.mall_id=='m_690'].latitude)
latimax = np.max(shopdata[shopdata.mall_id=='m_690'].latitude)
data = data.loc[:,['longitude','latitude']]
data = data.loc[(data.longitude<(0.00015+longmax)) & (data.longitude>(longmin-0.00015))]
data = data.loc[((data.latitude)<(0.00015+latimax)) & (data.latitude>(latimin-0.00015))]
print data.shape
x1_min, x2_min = np.min(data.values, axis=0)
x1_max, x2_max = np.max(data.values, axis=0)
plt.scatter(data['longitude'], data['latitude'], c='b', s=10, cmap=None, edgecolors='none')

shopdata = shopdata[shopdata.mall_id=='m_690']
shopdatalg = shopdata.loc[:,['longitude','latitude']]
# x1_min, x2_min = np.min(shopdatalg.values, axis=0)
# print x1_min, x2_min
# x1_max, x2_max = np.max(shopdatalg.values, axis=0)
# print x1_max, x2_max

plt.scatter(shopdata['longitude'], shopdata['latitude'], s=20, c = 'r',marker = 'o', edgecolors=None)
plt.xlim((x1_min+0.0005, x1_max+0.0005))
plt.ylim((x2_min+0.0005, x2_max+0.0005))
plt.grid(True)
plt.show()
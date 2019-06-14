import numpy as np
import pandas as pd
from datetime import datetime
from collections import *
import matplotlib.pyplot as plt

t0 = datetime.now()
def get_wifis(data):
    wifiDict = {
        'trade': [],
        'bssid': [],
        'strength': [],
        'connect': [],
        'shop': [],
    }
    res = 0
    for index, row in data.iterrows():
        for wifi in row.wifi_infos.split(';'):
            info = wifi.split('|')
            wifiDict['trade'].append(res)
            wifiDict['bssid'].append(info[0])
            wifiDict['strength'].append(info[1])
            wifiDict['connect'].append(info[2])
            wifiDict['shop'].append(row.shop_id)
        res += 1
    df = pd.DataFrame(wifiDict)
    wifi_info = list(df['bssid'])
    print 'origion shop num ', len(set(df['shop']))
    print 'origion tride num ', len(set(df['trade']))
    wifi_count = Counter(wifi_info)
    wifi_count = dict(wifi_count)
    wifi_count = {key: value for key, value in wifi_count.iteritems() if value > 3}
    wifi_ans = wifi_count.keys()
    df = df[df['bssid'].isin(wifi_ans)]
    print 'shop num ', len(set(df['shop']))
    print 'tride num ', len(set(df['trade']))
    df1 = pd.DataFrame.from_dict(wifi_count, orient='index')
    df1.columns = ['A']
    wifi_shops = {}
    for wifi in wifi_ans:
        wifi_shops[wifi] = len(set(df[df['bssid'] == wifi].shop))
    df2 = pd.DataFrame.from_dict(wifi_shops, orient='index')
    df2.columns = ['B']
    df3 = pd.concat([df1, df2], axis=1)
    df3['C'] = df3['A'] / df3['B']
    print 'get_wifis cost ', datetime.now()-t0
    print np.max(df3['B']),np.min(df3['B'])
    # counts = df3['B'].value_counts()  # select count
    # print counts
    return df3





# features = []
# for wifi in wifi_ans:
#     features += [wifi + 'C', wifi + 'K']
# features = wifi_ans + features
#
# df4 = pd.DataFrame(np.zeros((trade_num,len(features))),index=None,columns=features)
# j = 0
# y = []
# trade = df.iloc[0, 5]
# for row in df:
#     if row.trade == trade:
#         if row.bssid in features:
#             wifi_p = features.index(row.bssid)
#             df4.iloc[j, wifi_p] = row.strength
#             row.connect = 0 if row.connect == 'False'else 1
#             wifi_c = features.index(row.bssid + 'C')
#             df4.iloc[j, wifi_c] = row.connect
#             wifi_k = features.index(row.bssid + 'K')
#             df4.iloc[j, wifi_k] = df3.iloc[row.bssid,'C']
#             print df4
#             break





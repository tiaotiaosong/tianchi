import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
# Counter({'s_683671': 11, 's_683678': 8, 's_684245': 8, 's_683674': 6, 's_683821': 6, 's_683217': 6, 's_539726': 5, 's_682074': 5, 's_683053': 5, 's_683985': 5, 's_684235': 4, 's_684547': 4, 's_696563': 3, 's_684555': 3, 's_685227': 2, 's_809873': 2, 's_685058': 2, 's_2248290': 2, 's_681760': 2, 's_479529': 1, 's_683832': 1, 's_992598': 1, 's_1488950': 1, 's_470182': 1, 's_466068': 1})
# Counter({'s_685058': 33, 's_683671': 19, 's_683678': 17, 's_683985': 16, 's_683674': 15, 's_679110': 14, 's_684547': 13, 's_683821': 12, 's_683832': 11, 's_809873': 11, 's_662370': 11, 's_684245': 11, 's_684555': 11, 's_684235': 9, 's_683053': 9, 's_2248290': 9, 's_461325': 8, 's_685227': 7, 's_3382317': 7, 's_539726': 7, 's_683217': 7, 's_698556': 6, 's_521874': 6, 's_472478': 6, 's_466068': 6, 's_694816': 5, 's_682074': 5, 's_710741': 5, 's_991943': 5, 's_3421010': 5, 's_696563': 4, 's_635423': 4, 's_712117': 4, 's_3991685': 4, 's_473743': 4, 's_3167757': 4, 's_592650': 3, 's_479529': 3, 's_3446229': 3, 's_3621982': 3, 's_3101804': 3, 's_3140371': 3, 's_509650': 3, 's_774995': 3, 's_465792': 3, 's_505726': 3, 's_503249': 2, 's_505087': 2, 's_622845': 2, 's_2230096': 2, 's_769794': 2, 's_739312': 2, 's_470352': 2, 's_681760': 2, 's_470182': 2, 's_503457': 1, 's_3257365': 1, 's_992598': 1, 's_696015': 1, 's_696567': 1, 's_1093389': 1, 's_709290': 1, 's_546469': 1, 's_528544': 1, 's_479486': 1, 's_735757': 1, 's_3812635': 1, 's_1114306': 1, 's_604425': 1, 's_1488950': 1, 's_3832595': 1, 's_1444403': 1})

owifi_strength = {}
owifi_connect ={}
owifi_num = {}
odata = pd.read_csv('m_7800data.csv')
for line, row in odata.iterrows():
    wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
    for i in wifi_list:
        i[2] = 0 if i[2] == 'false'else 1
        if i[0] not in owifi_strength:
            owifi_strength[i[0]] = int(i[1])
            owifi_num[i[0]] = 1
            owifi_connect[i[0]] = i[2]
        else:
            owifi_strength[i[0]] += int(i[1])
            owifi_num[i[0]] += 1
            owifi_connect[i[0]] += i[2]
owifi_strength = pd.DataFrame.from_dict(owifi_strength,orient='index')
owifi_strength.columns = ['ostrength']
owifi_num = pd.DataFrame.from_dict(owifi_num,orient='index')
owifi_num.columns = ['onum']
owifi_connect = pd.DataFrame.from_dict(owifi_connect,orient='index')
owifi_connect.columns = ['oconnect']
oout = pd.concat([owifi_strength,owifi_num,owifi_connect],axis=1)
oout['oAverage'] = oout['ostrength'] / oout['onum']


data = odata[odata.shop_id=='s_683671']

print len(data)
long = data['longitude']
# plt.scatter(data['longitude'], data['latitude'], s=20, c = 'b',marker = 'o', edgecolors=None)
# plt.scatter(109.265016, 35.258217, s=20, c = 'r',marker = 'o', edgecolors=None)
plt.hist(long, bins=100, color='steelblue', normed=True )
# kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data['longitude'])
# dens = kde.score_samples(X_plot)
plt.show()
wifi_strength = {}
wifi_connect ={}
wifi_num = {}
for line, row in data.iterrows():
    wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
    for i in wifi_list:
        i[2] = 0 if i[2] == 'false'else 1
        if i[0] not in wifi_strength:
            wifi_strength[i[0]] = int(i[1])
            wifi_num[i[0]] = 1
            wifi_connect[i[0]] = i[2]
        else:
            wifi_strength[i[0]] += int(i[1])
            wifi_num[i[0]] += 1
            wifi_connect[i[0]] += i[2]

wifi_strength = pd.DataFrame.from_dict(wifi_strength,orient='index')
wifi_strength.columns = ['strength']
wifi_num = pd.DataFrame.from_dict(wifi_num,orient='index')
wifi_num.columns = ['num']
wifi_connect = pd.DataFrame.from_dict(wifi_connect,orient='index')
wifi_connect.columns = ['connect']
out = pd.concat([wifi_strength,wifi_num,wifi_connect],axis=1)
out['Average'] = out['strength'] / out['num']
out['occupy'] = out['num'] / len(data)
out3 = pd.concat([out,oout],axis=1,join_axes=[out.index])
out3['wifiocc'] = out3['num']/out3['onum']
out3.to_csv ("out.csv" , index=True,header=True,encoding = "utf-8")
print out3
# Sdata = pd.DataFrame(counts)
# wifi_sta = pd.Series()
# l = []
# wifi_dict = {}
# for index, row in data.iterrows():
#     wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
#     for i in wifi_list:
#         row[i[0]] = int(i[1])
#         if i[0] not in wifi_dict:
#             wifi_dict[i[0]] = 1
#         else:
#             wifi_dict[i[0]] += 1
#     l.append(row)
# delate_wifi=[]
# for i in wifi_dict:
#     if wifi_dict[i]<20:
#         delate_wifi.append(i)
#
# for line, row in data.iterrows():
#     if row[1] not in wifi_sta.index:
#         wifi_sta[row[1]] = pd.Series()
#         wifi_sta[row[1]]['trade'] = 1
#     wifi_sta[row[1]]['trade'] += 1
#     wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
#     for i in wifi_list:
#         if i[0] not in wifi_sta[row[1]].index and i[0] in delate_wifi:
#             wifi_sta[row[1]][i[0]] = 1
#             wifi_sta[row[1]][i[0]+'S'] = int(i[1])
#         else:
#             wifi_sta[row[1]][i[0]] += 1
#             wifi_sta[row[1]][i[0] + 'S'] += int(i[1])
#
# # for row in wifi_dict:
# #     row[] b_\w+S
#
# print wifi_dict
# wifi_dict.to_csv(r"wifi_dict.csv", index=None)

# Counter({'s_683671': 11, 's_683678': 8, 's_684245': 8, 's_683674': 6, 's_683821': 6, 's_683217': 6, 's_539726': 5, 's_682074': 5, 's_683053': 5, 's_683985': 5, 's_684235': 4, 's_684547': 4, 's_696563': 3, 's_684555': 3, 's_685227': 2, 's_809873': 2, 's_685058': 2, 's_2248290': 2, 's_681760': 2, 's_479529': 1, 's_683832': 1, 's_992598': 1, 's_1488950': 1, 's_470182': 1, 's_466068': 1})
# Counter({'s_685058': 33, 's_683671': 19, 's_683678': 17, 's_683985': 16, 's_683674': 15, 's_679110': 14, 's_684547': 13, 's_683821': 12, 's_683832': 11, 's_809873': 11, 's_662370': 11, 's_684245': 11, 's_684555': 11, 's_684235': 9, 's_683053': 9, 's_2248290': 9, 's_461325': 8, 's_685227': 7, 's_3382317': 7, 's_539726': 7, 's_683217': 7, 's_698556': 6, 's_521874': 6, 's_472478': 6, 's_466068': 6, 's_694816': 5, 's_682074': 5, 's_710741': 5, 's_991943': 5, 's_3421010': 5, 's_696563': 4, 's_635423': 4, 's_712117': 4, 's_3991685': 4, 's_473743': 4, 's_3167757': 4, 's_592650': 3, 's_479529': 3, 's_3446229': 3, 's_3621982': 3, 's_3101804': 3, 's_3140371': 3, 's_509650': 3, 's_774995': 3, 's_465792': 3, 's_505726': 3, 's_503249': 2, 's_505087': 2, 's_622845': 2, 's_2230096': 2, 's_769794': 2, 's_739312': 2, 's_470352': 2, 's_681760': 2, 's_470182': 2, 's_503457': 1, 's_3257365': 1, 's_992598': 1, 's_696015': 1, 's_696567': 1, 's_1093389': 1, 's_709290': 1, 's_546469': 1, 's_528544': 1, 's_479486': 1, 's_735757': 1, 's_3812635': 1, 's_1114306': 1, 's_604425': 1, 's_1488950': 1, 's_3832595': 1, 's_1444403': 1})


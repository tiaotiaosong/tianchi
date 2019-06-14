import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()
#
users = []
abnormal = []
# # wifis = ['long','lati']
wifis = []
shops = []
data = pd.read_csv('m_690data.csv')
for array in data.values:
    if array[1] not in shops:
        shops.append(array[1])
    array = array[5].split(';')
    for line in array:
        line=line.split('|')
        if line[0] not in wifis:
            wifis.append(line[0])
wifi_g = np.zeros((1, len(wifis)))
for array in data.values:
    wifi = array[5].split(';')
    for line in wifi:
        wifidata=line.split('|')
        wifi_p = wifis.index(wifidata[0])
        wifi_g[0,wifi_p] += 1
df = pd.DataFrame(wifi_g,index=['num'],columns=wifis)
# dff = df.sort_values(by='num',axis=1,ascending=False)
df=df.T
df=df[df.num>3]
print df.index         #selected wifi ID

print np.sum(df['num'])                    #the number of trade
counts = df['num'].value_counts()      #select count
print sum(counts.values)     #the number of wifi
# counts.hist(bins=100)
# plt.show()

###########user purchase frequency
# for array in data.values:
#     if array[0] not in users:
#         users.append(array[0])
# user_g = np.zeros((1, len(users)))
# for array in data.values:
#     user_p = users.index(array[0])
#     user_g[0, user_p] += 1
# Udf = pd.DataFrame(user_g, index=['num'], columns=users)
# Udf=Udf.T
# Udf=Udf[Udf.num>20]
# dff = Udf.sort_values(by='num',axis=0,ascending=False)
# print dff

# for array in data.values:
#     if array[0] in dff.index:
#         if array[1]=='s_298312':
#             abnormal.append(array[2])
# # print abnormal
# # print len(abnormal)
# print dff




import csv
import numpy as np
import pandas as pd


def createDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in dataDict.iteritems():
            csvWriter.writerow([k, v])
        csvFile.close()
#
shopswifi = {}
dict = {}
data = pd.read_csv('m_1263data.csv')
for array in data.values:
    if array[1] not in shopswifi:
        shopswifi[array[1]]={}
    wifi = array[5].split(';')
    for line in wifi:
        line = line.split('|')
        if line[0] not in shopswifi[array[1]]:
            shopswifi[array[1]][line[0]] = 1
        else:
            shopswifi[array[1]][line[0]] += 1
for array in shopswifi:
    dict[array]= sorted(shopswifi[array].iteritems(), key=lambda d: d[1], reverse=True)

createDictCSV("shopswifi.csv", dict)








# user_g = np.zeros((1, len(shops)))
# for array in data.values:
#     user_p = shops.index(array[0])
#     user_g[0, user_p] += 1
# Udf = pd.DataFrame(user_g, index=['num'], columns=users)
# Udf=Udf.T
# Udf=Udf[Udf.num>20]
# dff = Udf.sort_values(by='num',axis=0,ascending=False)
# # print dff
#
# for array in data.values:
#     if array[0] in dff.index:
#         if array[1]=='s_298312':
#             abnormal.append(array[2])
# # print abnormal
# # print len(abnormal)
# print dff
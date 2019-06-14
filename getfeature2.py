import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime


t0 = datetime.now()
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()

users = []
abnormal = []
wifis = []
shops = []
data = pd.read_csv('m_690data.csv')
shopdata = pd.read_csv('ccf_first_round_shop_info.csv',header=0)
longmin = np.min(shopdata[shopdata.mall_id=='m_690'].longitude)
longmax = np.max(shopdata[shopdata.mall_id=='m_690'].longitude)
longmean = np.mean(shopdata[shopdata.mall_id=='m_690'].longitude)
latimin = np.min(shopdata[shopdata.mall_id=='m_690'].latitude)
latimax = np.max(shopdata[shopdata.mall_id=='m_690'].latitude)
latimean = np.mean(shopdata[shopdata.mall_id=='m_690'].latitude)
print longmax-longmin
print latimax-latimin

for array in data.values:
    if array[1] not in shops:
        shops.append(array[1])
    array = array[5].split(';')
    for line in array:
        line=line.split('|')
        if line[0] not in wifis:
            wifis.append(line[0])
print data.values.shape[0]

wifi_g = np.zeros((1, len(wifis)))
for array in data.values:
    long = float(array[3])
    lati = float(array[4])
    if (long - longmean) > 0.0015 or (lati - latimean) > 0.0015:
        abnormal.append(array)
        continue
    wifi = array[5].split(';')
    for line in wifi:
        wifidata=line.split('|')
        wifi_p = wifis.index(wifidata[0])
        wifi_g[0,wifi_p] += 1
df = pd.DataFrame(wifi_g,index=['num'],columns=wifis)
df=df.T
df=df[df.num>3]
wifis=['long','lati']+list(df.index)
print len(wifis)
dnf = pd.DataFrame(np.zeros((data.values.shape[0],len(wifis))),index=None,columns=wifis)
dsf = pd.DataFrame(np.zeros((data.values.shape[0],len(shops))),index=None,columns=shops)
i = 0
for array in data.values:
    long = float(array[3])
    lati = float(array[4])
    if (long-longmean) >0.0015  or (lati-latimean) >0.0015:
        abnormal.append(array)
        continue
    shop_p = shops.index(array[1])
    dsf.iloc[i, shop_p] = 1
    dnf.iloc[i,0] = (long - longmin) / (latimax - latimin)
    dnf.iloc[i,1] = (lati-latimin)/(latimax-latimin)
    wifi = array[5].split(';')
    for line in wifi:
        wifidata=line.split('|')
        if wifidata[0] in wifis:
            wifi_p = wifis.index(wifidata[0])
            dnf.iloc[i,wifi_p] = (100+float(wifidata[1]))/80
    i +=1

dnf.to_csv ("testfoo.csv" , index=False,header=True,encoding = "utf-8")
dsf.to_csv ("testshop.csv" , index=False,header=True,encoding = "utf-8")
print dnf.values.shape
print dsf.values.shape
print datetime.now()-t0
print len(abnormal)




import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from getwifis import get_wifis
from sklearn.decomposition import PCA
t0 = datetime.now()
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()

shopdata = pd.read_csv('ccf_first_round_shop_info.csv',header=0)
longmin = np.min(shopdata[shopdata.mall_id=='m_7800'].longitude)
longmax = np.max(shopdata[shopdata.mall_id=='m_7800'].longitude)
latimin = np.min(shopdata[shopdata.mall_id=='m_7800'].latitude)
latimax = np.max(shopdata[shopdata.mall_id=='m_7800'].latitude)

users = []
abnormal = []
shops = []

data = pd.read_csv('m_7800data.csv')
print data.shape
data = data.loc[(data.longitude<(0.00015+longmax)) & (data.longitude>(longmin-0.00015))]
data = data.loc[((data.latitude)<(0.00015+latimax)) & (data.latitude>(latimin-0.00015))]
print data.shape

for array in data.values:
    if array[1] not in shops:
        shops.append(array[1])
df = get_wifis(data)
wifi_ans=list(df.index)
features = []
for wifi in wifi_ans:
    features += [ wifi + 'C']
    # features += [wifi + 'C', wifi + 'K']
features = wifi_ans + features
wifis=['long','lati']+list(features)
print len(wifis)
# print wifis
dnf = pd.DataFrame(np.zeros((data.values.shape[0],len(wifis))),index=None,columns=wifis)
dsf = pd.DataFrame(np.zeros((data.values.shape[0],len(shops))),index=None,columns=shops)
ds = pd.DataFrame(np.zeros((data.values.shape[0],1)),index=None,columns=['shop'])
i = 0
for array in data.values:
    long = float(array[3])
    lati = float(array[4])
    shop_p = shops.index(array[1])
    dsf.iloc[i, shop_p] = 1
    ds.iloc[i, 0] = shop_p
    dnf.iloc[i,0] = (long - longmin) / (latimax - latimin)
    dnf.iloc[i,1] = (lati-latimin)/(latimax-latimin)
    wifi = array[5].split(';')
    for line in wifi:
        wifidata=line.split('|')
        if wifidata[0] in wifis:
            wifi_p = wifis.index(wifidata[0])
            dnf.iloc[i,wifi_p] =int(wifidata[1])+100
            wifidata[2] = 0 if wifidata[2] == 'false'else 1
            wifi_c = features.index(wifidata[0] + 'C')
            dnf.iloc[i, wifi_c] = wifidata[2]
            # wifi_k = features.index(wifidata[0] + 'K')
            # dnf.iloc[i, wifi_k] = df.loc[wifidata[0],'C']
    i +=1
print dnf.shape
# scaler = preprocessing.MinMaxScaler()
# dnf = pd.DataFrame(scaler.fit_transform(dnf.values), columns=wifis)
# pca = PCA(n_components = 0.98,svd_solver = 'full', whiten=True, random_state=0)
# dnf = pd.DataFrame(pca.fit_transform(dnf))
print dnf.shape
print dnf
dnf.to_csv ("testfoo.csv" , index=False,header=True,encoding = "utf-8")
dsf.to_csv ("testshop.csv" , index=False,header=True,encoding = "utf-8")
ds.to_csv ("testshop1.csv" , index=False,header=True,encoding = "utf-8")

print dnf.values.shape
print datetime.now()-t0


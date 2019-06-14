import csv
import numpy as np
import pandas as pd


def createDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in dataDict.iteritems():
            csvWriter.writerow([k, v])
        csvFile.close()

shopswifi = {}
dict = {}
dict1 ={}
wifis=[]
data = pd.read_csv('m_7800data.csv')
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
    dict1[array] = {k: v for k,v in shopswifi[array].iteritems() if v > 0}
for array in dict1:
    dict[array]= sorted(dict1[array].iteritems(), key=lambda d: d[1], reverse=True)
    wifis += dict1[array].keys()
createDictCSV("shopswifi.csv", dict)
print len(set(wifis))
shopswifinum = {}
for array in dict:
    shopswifinum[array] = len(dict[array])
print shopswifinum
df = pd.DataFrame.from_dict(shopswifinum,orient='index')
df.columns = ['a']
print df
print np.min(df.a)
def getwifis():
    return set(wifis)
import csv
import numpy as np
import pandas as pd
from collections import *
import matplotlib.pyplot as plt

def createDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for k, v in dataDict.iteritems():
            csvWriter.writerow([k, v])
        csvFile.close()


def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()
wifis = {}
dict = {}
dict1 = {}
data = pd.read_csv('m_1263data.csv')
for array in data.values:
    wifi = array[5].split(';')
    for line in wifi:
        line=line.split('|')
        if line[0] not in wifis:
            wifis[line[0]]={}

for array in data.values:
    wifi = array[5].split(';')
    for line in wifi:
        line = line.split('|')
        if array[1] not in wifis[line[0]]:
            wifis[line[0]][array[1]] = 1
        else:
            wifis[line[0]][array[1]] += 1
# print wifis
for array in wifis:
    dict[array]= sorted(wifis[array].iteritems(), key=lambda d: d[1], reverse=True)
dict= sorted(dict.iteritems(), key=lambda d: d[0], reverse=False)
createListCSV("wifisshop.csv", dict)
wifishopnum = {}
for array in dict:
    wifishopnum[array[0]] = len(array[1])
print wifishopnum
df = pd.DataFrame.from_dict(wifishopnum,orient='index')
df.columns = ['a']
counts = df['a'].value_counts()
print counts
counts.hist(bins=100)
plt.show()
print df
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def createDictCSV(fileName="", dataDict={}):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['malls', 'lg'])
        for k,v in dataDict.iteritems():
            csvWriter.writerow([k,v])
        csvFile.close()
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow('da')
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()
data = pd.read_csv('ccf_first_round_shop_info.csv', header=0)
malls = {}
malls2 = {}
mallsshops = {}

for array in data.values:
    longitude=array[2]
    if array[-1] not in malls:
        malls[array[-1]] = longitude
        mallsshops[array[-1]] = 1
    else:
        malls[array[-1]] += longitude
        mallsshops[array[-1]] += 1
for line in malls:
    malls[line] =malls[line]/mallsshops[line]

for array in data.values:
    latitude=array[3]
    if array[-1] not in malls2:
        malls2[array[-1]] = latitude
    else:
        malls2[array[-1]] += latitude
for line in malls2:
    malls2[line] =malls2[line]/mallsshops[line]

for line in malls:
    malls[line] =(malls[line], malls2[line])
mallss=malls.values()
createListCSV("malls.csv", mallss)
print mallss
createDictCSV("mallsposition.csv", malls)
print malls
print mallsshops
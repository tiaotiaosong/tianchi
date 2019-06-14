import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

print datetime.now()
# def createListCSV(fileName="", dataList=[]):
#     with open(fileName, "wb") as csvFile:
#         csvWriter = csv.writer(csvFile)
#         for data in dataList:
#             csvWriter.writerow(data)
#         csvFile.close()

traindata = np.vstack((pd.read_csv('wifidata0.csv',header=0).values,pd.read_csv('wifidata1.csv',header=0).values,
                      pd.read_csv('wifidata2.csv',header=0).values,pd.read_csv('wifidata3.csv',header=0).values))
                     # pd.read_csv('wifidata4.csv', header=0).values, pd.read_csv('wifidata5.csv', header=0).values))
print traindata.shape

trainlabel = pd.read_csv('shopdata.csv',header=0).values[:12000,:]
print trainlabel.shape

testdata = pd.read_csv('wifidata6.csv',header=0)
testlabel = pd.read_csv('shopdata.csv',header=0).values[18000:21000,:]

##########
clf = KNeighborsClassifier(n_neighbors=5,weights='uniform')
clf.fit(traindata, trainlabel)
test = clf.predict(testdata.values)
accuracy = np.trace(np.dot(np.array(test),testlabel.T))/3000

print clf.score(traindata,trainlabel)
print clf.score(testdata,testlabel)
print accuracy
print datetime.now()
######## trainscore 1.0  testscore0.827666666667    samples 12000 n_neighbors=5,weights='distance'
########trainscore 1.0   testscore 0.832666666667   samples 12000 n_neighbors=3,weights='distance'
######## hao nei cun , man 20min
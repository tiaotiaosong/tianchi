import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

t0 = datetime.now()
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close()

X = pd.read_csv('testfoo.csv')###,header=None
y = pd.read_csv('testshop.csv')
print (X.shape)
print (y.shape)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(200,150), random_state=1,max_iter=300)
#
# scores = cross_val_score(clf, X, y,cv=4 )
# print (scores)
# print (datetime.now()-t0)
skf = StratifiedKFold(n_splits=3)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data.values, label.values, test_size=0.4, random_state=0)
########## best0.855 on hidden_layer_sizes=(300 250)  from(100-300,100-300) on 12000samples


clf.fit(traindata, trainlabel)
test = clf.predict(testdata)
# accuracy= np.trace(np.dot(np.array(test),testlabel.T))/(testdata.shape[1])

print clf.score(traindata,trainlabel)
print clf.score(testdata,testlabel)
# print np.max(np.dot(np.array(test),testlabel.T))
print datetime.now()-t0
########trainscore 0.93075 testscore 0.832333333333  samples 12000

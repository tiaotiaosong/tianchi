import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

t0 = datetime.now()
# def createListCSV(fileName="", dataList=[]):
#     with open(fileName, "wb") as csvFile:
#         csvWriter = csv.writer(csvFile)
#         for data in dataList:
#             csvWriter.writerow(data)
#         csvFile.close()
#
X = pd.read_csv('testfoo.csv')#,header=None
print X.shape

# X =  pd.read_csv('testfoo.csv').iloc[:,2:]

y = pd.read_csv('testshop1.csv')
y = np.ravel(y)

# oob_score

gsearch1 = RandomForestClassifier(min_samples_leaf=1,max_features='sqrt',criterion="gini",n_estimators=190,
                                               min_samples_split=6,oob_score=True,class_weight=None),


gsearch1.fit(X.values, y)
print score()
print
# y = np.array(y)
# y=y.T
# print y
# y=y.tolist()
# print y
# y=np.array(y)[0]
# print type(y),y.shape

# y = np.array(y.values).T
##########
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
# clf = RandomForestClassifier()
# clf.fit(train_x, train_y)
# print clf.score(train_x,train_y)
# print clf.score(test_x,test_y)
# print clf.predict(test_x)[:100]
# print test_y[:100]

# clf = RandomForestClassifier()
# scores = cross_val_score(clf, X, y,cv=4 )
# print scores
print datetime.now()-t0
#[ 0.64638091  0.63826334  0.63933089  0.62753534] without lg
#[ 0.66336433  0.64232096  0.65095879  0.63839377]   have lg
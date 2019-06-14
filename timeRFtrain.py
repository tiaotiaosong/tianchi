# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

@author: 麦芽的香气
"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from datetime import datetime
from collections import *
from sklearn.ensemble import RandomForestClassifier

t0 = datetime.now()
df=pd.read_csv('ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv('ccf_first_round_shop_info.csv')
test=pd.read_csv(u'AB榜测试集-evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
# df['time_stamp']=pd.to_datetime(df['time_stamp'])
traindata=pd.concat([df,test])
mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
for mall in mall_list:
    if mall =='m_7800':
        train1 = traindata[traindata.mall_id == mall].reset_index(drop=True)
        l = []
        wifi_dict = {}
        for index, row in train1.iterrows():
            r = {}
            t = (row['time_stamp'].split())[0].split('-')
            r['day'] = int(t[2])
            r['time'] = pd.to_datetime(row['time_stamp']).weekday()
            r['hour'] = pd.to_datetime(row['time_stamp']).hour


            wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
            for i in wifi_list:
                r[i[0]] = int(i[1])
                if i[0] not in wifi_dict:
                    wifi_dict[i[0]] = 1
                else:
                    wifi_dict[i[0]] += 1
            l.append(r)
        delate_wifi = []
        for i in wifi_dict:
            if wifi_dict[i] < 20:
                delate_wifi.append(i)
        m = []
        for row in l:
            new = {}
            for n in row.keys():
                if n not in delate_wifi:
                    new[n] = row[n]
            m.append(new)
        train1 = pd.concat([train1, pd.DataFrame(m)], axis=1)
        df_train = train1[train1.shop_id.notnull()]
        df_test = train1[train1.shop_id.isnull()]

        # shopdic = df_train['shop_id'].value_counts()
        # shoplist = [key for key, value in shopdic.iteritems() if value > 10]
        # df_train = df_train[df_train['shop_id'].isin(shoplist)]2017-08-06 21:20
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train['shop_id'].values))
        df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
        timepoint = pd.to_datetime('2017-08-25 00:00')
        df_train['time_stamp'] = pd.to_datetime(df_train['time_stamp'])
        train_x = df_train[df_train.time_stamp < timepoint]
        test_x = df_train[df_train.time_stamp >= timepoint]
        train_y = train_x['label']
        test_y = test_x['label']
        # print test_x'hour','time'
        feature = [x for x in train1.columns if
                   x not in ['user_id', 'label','row_id', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos',
                             ]]
        df_train_1 = train_x[feature]
        df_train_1 = df_train_1.where(df_train_1.notnull(), -100)
        print df_train_1
        # train_x, test_x, train_y, test_y = train_test_split(df_train_1.values, df_train['label'].values, test_size=0.2, random_state=0)
        df_test_1 = test_x[feature]
        df_test_1 = df_test_1.where(df_test_1.notnull(), -100)

        # del train_x
        # del test_x
        # gc.collect()
        # xgbtrain = xgb.DMatrix(train_x, label=train_y)
        # clf = RandomForestClassifier(n_estimators=190, min_samples_leaf=1, max_features='sqrt', criterion="gini",
        #                              min_samples_split=6, oob_score=True, class_weight="balanced")
        clf = RandomForestClassifier(n_estimators=190, min_samples_leaf=1, max_features='sqrt', criterion="gini",
                                                                  min_samples_split=6, oob_score=True, class_weight="balanced")
        clf.fit(df_train_1, train_y)

        print clf.score(df_train_1, train_y)
        print clf.score(df_test_1, test_y)
        # print df_train_1.columns==df_test_1.columns
        # print df_train_1.columns

        print df_train_1.shape
        print df_test_1.shape
        # ggx = model.predict(test_x)
        # y_pred = np.argmax(ggx, axis=1)
        # j=0
        # for x in range(len(test_y)):
        #     if  y_pred[x]==test_y[x]:
        #         j +=1
        #
        # print j/float(len(test_y))
        # print y_pred
        # print test_y
        print 'time cost',datetime.now() - t0
        # ypred = model.predict(data)
        # df_test['label'] = model.predict(xgbtest)
        # df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))
        # r = df_test[['row_id', 'shop_id']]
        # result = pd.concat([result, r])
        # result['row_id'] = result['row_id'].astype('int')
        # result.to_csv('sub.csv', index=False)

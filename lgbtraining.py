# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

@author: 麦芽的香气
"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
# import xgboost as xgb
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from datetime import datetime
from collections import *

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
            # t = (row['time_stamp'].split())[1].split(':')
            # r['min'] = int(t[1])
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
            if wifi_dict[i] < 10:
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

        shopdic = df_train['shop_id'].value_counts()
        shoplist = [key for key, value in shopdic.iteritems() if value > 10]
        df_train = df_train[df_train['shop_id'].isin(shoplist)]
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train['shop_id'].values))
        df_train['label'] = lbl.transform(list(df_train['shop_id'].values))

        num_class = df_train['label'].max() + 1
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_error',
            'num_leaves': 31,
            'num_class': num_class,
            'learning_rate': 0.05,
            'min_data_in_leaf':20,
            'tree_learner':'serial',
            # 'feature_fraction':0.6,
            # # 'feature_fraction_seed':2,
            # 'bagging_fraction':0.6,
            # 'lambda_l1':0.01,
            # 'lambda_l2':0.01,
            'max_depth':12,
        }
        # params = {
        #     'task': 'train',
        #     'boosting_type': 'gbdt',
        #     'objective': 'multiclassova',
        #     'metric': 'multi_error',
        #     'num_leaves': 31,
        #     'num_class': num_class,
        #     'learning_rate': 0.05,
        #     'min_data_in_leaf':20,
        #     'tree_learner':'serial',
        #     # 'feature_fraction':0.6,
        #     # # 'feature_fraction_seed':2,
        #     # 'bagging_fraction':0.6,
        #     # 'lambda_l1':0.01,
        #     # 'lambda_l2':0.01,
        #     'max_depth':12,
        # }
        timepoint = pd.to_datetime('2017-08-27 00:00')
        df_train['time_stamp'] = pd.to_datetime(df_train['time_stamp'])
        train_x = df_train[df_train.time_stamp < timepoint]
        test_x = df_train[df_train.time_stamp >= timepoint]
        train_y = train_x['label']
        test_y = test_x['label']
        # print test_x
        feature = [x for x in train1.columns if
                   x not in ['user_id', 'label','row_id', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]
        df_train_1 = train_x[feature]
        df_train_1 = df_train_1.where(df_train_1.notnull(), -100)

        # train_x, test_x, train_y, test_y = train_test_split(df_train_1.values, df_train['label'].values, test_size=0.2, random_state=0)
        df_test_1 = test_x[feature]
        df_test_1 = df_test_1.where(df_test_1.notnull(), -100)

        del train_x
        del test_x
        gc.collect()
        # xgbtrain = xgb.DMatrix(train_x, label=train_y)
        train_data = lgb.Dataset(df_train_1, label=train_y,max_bin=255)
        # xgbtest = xgb.DMatrix(test_x, label=test_y)
        test_data = lgb.Dataset(df_test_1, label=test_y,max_bin=255, reference=train_data)
        #  valid_sets= [test_data, train_data], valid_names=['test', 'train'] ,
        model = lgb.train(params, train_data, num_boost_round=200, valid_sets= [test_data, train_data], valid_names=['test', 'train'] ,
                       init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=15,
                       evals_result=None, verbose_eval=10,
                       callbacks=None)
        print (Counter(train_y))
        print (Counter(test_y))

        print (df_train_1.shape)
        print (df_test_1.shape)
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
        print ('time cost',datetime.now() - t0)
        # ypred = model.predict(data)
        # df_test['label'] = model.predict(xgbtest)
        # df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))
        # r = df_test[['row_id', 'shop_id']]
        # result = pd.concat([result, r])
        # result['row_id'] = result['row_id'].astype('int')
        # result.to_csv('sub.csv', index=False)

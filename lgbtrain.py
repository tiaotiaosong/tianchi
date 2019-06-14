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

t0 = datetime.now()
df=pd.read_csv('ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv('ccf_first_round_shop_info.csv')
test=pd.read_csv(u'AB榜测试集-evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
# df['time_stamp']=pd.to_datetime(df['time_stamp'])
traindata=pd.concat([df,test])
mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
j=0
for mall in mall_list:
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
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    # print np.min(df_train['label'].values)
    # print (np.max(df_train['label'].values)
    # print df_train['label'].value_counts()
    num_class = df_train['label'].max() + 1
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_error',
        'num_leaves': 31,
        'num_class': num_class,
        'learning_rate': 0.05,
        'min_data_in_leaf': 20,
        'tree_learner': 'serial',
        # 'feature_fraction':0.6,
        # # 'feature_fraction_seed':2,
        # 'bagging_fraction':0.6,
        # 'lambda_l1': 0.01,
        # 'lambda_l2': 0.01,
        'max_depth': 11,
    }
    feature = [x for x in train1.columns if
               x not in ['user_id', 'label', 'shop_id', 'time_stamp', 'mall_id', 'wifi_infos']]
    df_train_1 = df_train[feature]
    df_train_1 = df_train_1.where(df_train_1.notnull(), -100)
    # train_x, test_x, train_y, test_y = train_test_split(df_train_1.values, df_train['label'].values, test_size=0.2, random_state=1)
    df_test_1 = df_test[feature]
    df_test_1 = df_test_1.where(df_test_1.notnull(), -100)
    # xgbtrain = xgb.DMatrix(train_x, label=train_y)
    df_train_1 = lgb.Dataset(df_train_1, label=df_train['label'], max_bin=255)
    # xgbtest = xgb.DMatrix(test_x, label=test_y)
    # valid_sets = [df_train_1], valid_names = ['train'],early_stopping_rounds=10,
    # valid_sets = [train_data], valid_names = ['train'],early_stopping_rounds=10,
    model = lgb.train(params, df_train_1, num_boost_round=200,
                      init_model=None, feature_name='auto', categorical_feature='auto',verbose_eval=False,
                      evals_result=None,callbacks=None)
    df_test['label'] = np.argmax(model.predict(df_test_1.values), axis=1)
    df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))
    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv('sub.csv', index=False)
    j=j+1
    print (j)
print ('time cost',datetime.now() - t0)

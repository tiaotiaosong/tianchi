# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

@author: 麦芽的香气
"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lgbtraining as lgb
df=pd.read_csv('ccf_first_round_user_shop_behavior.csv')#读入训练数据一
shop=pd.read_csv('ccf_first_round_shop_info.csv')#读入训练数据二
test=pd.read_csv(u'AB榜测试集-evaluation_public.csv')#读入测试数据
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id') #合并训练数据二的mall_id到训练数据一
df['time_stamp']=pd.to_datetime(df['time_stamp']) #将字符串转化为时间格式
train=pd.concat([df,test])#将训练数据与测试数据上下合并
mall_list=list(set(list(shop.mall_id))) #罗列出所有商场
result=pd.DataFrame() #定义输出数据框
j=0
for mall in mall_list:  #循环每个商场
    train1=train[train.mall_id==mall].reset_index(drop=True)         #选出该商场的所有训练测试数据
    l=[]
    wifi_dict = {}  #列出所有wifi，形成一个字典
    for index,row in train1.iterrows():  #迭代该商场的所有训练测试数据的每一行
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]  #分割每行数据中出现的wifi
        for i in wifi_list:   #对wifi出现的次数计数，并存入wifi_dict字典
            r[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(r)    
    delate_wifi=[]
    for i in wifi_dict:   #删除所有出现次数少于5次的wifi
        if wifi_dict[i]<15:
            delate_wifi.append(i)
    m=[]
    for row in l:   #删除wifi之后的训练测试数据
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)    #左右合并原始训练测试数据与新生成训练测试数据
    df_train=train1[train1.shop_id.notnull()]  #根据有shop_id，分割数据为训练数据，
    df_test=train1[train1.shop_id.isnull()]  #根据无shop_id，分割数据为测试数据，
    lbl = preprocessing.LabelEncoder()      #对标签进行编码为1—N
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1
    # 训练函数参数设定
    params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 9,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
    #从数据中去掉多余的列
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]
    df_train_1 = df_train[feature]          #训练数据对null值进行填充-100
    df_train_1 = df_train_1.where(df_train_1.notnull(), -100)
    xgbtrain = xgb.DMatrix(df_train_1, df_train['label'])
    df_test_1 = df_test[feature]           #测试数据对null值进行填充-100
    df_test_1 = df_test_1.where(df_test_1.notnull(), -100)
    xgbtest = xgb.DMatrix(df_test_1)
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds=100
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)   #训练模型
    df_test['label']=model.predict(xgbtest)        #预测
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))   #对标签1-N转化为shop_id
    r=df_test[['row_id','shop_id']]  #选出r
    result=pd.concat([result,r])
    j = j + 1
    print j
result['row_id']=result['row_id'].astype('int')
result.to_csv('sub.csv',index=False)

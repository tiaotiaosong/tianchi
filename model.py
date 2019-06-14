# -*- coding:utf-8 -*-
__author__ = 'ZLZ'
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA
import time
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


class SHDW(object):
    def __init__(self):

        self.C = None

    def generate_train_data(self, data):
        """
        对wifi信息生成数据特征,筛选部分特征，并对特征做归一化处理
        :return:
        """
        start = time.time()
        wifiDict = {
            'index': [],
            'bssid': [],
            'strength': [],
            'connect': []
        }
        res = 0
        shop_bssid = defaultdict(lambda: defaultdict(lambda: 0))
        wifi_shop_num = defaultdict(lambda: defaultdict(lambda: 0))
        wifi_number = []
        for index, row in data.iterrows():
            shop = row[1]
            ans = 0
            for wifi in row.wifi_infos.split(';'):
                ans += 1
                info = wifi.split('|')
                wifiDict['index'].append(res)
                wifiDict['bssid'].append(info[0])
                wifiDict['connect'].append(info[2])
                wifiDict['strength'].append(info[1])
                shop_bssid[shop][info[0]] += 1
                wifi_shop_num[info[0]][shop] += 1
            wifi_number.append(ans)
            res += 1
        df = pd.DataFrame(wifiDict)
        # wifi_s = Counter(df.strength)
        # print wifi_s



        # 对我出现的次数进行统计,并去掉出现次数少的wifi
        wifi_info = list(df['bssid'])
        wifi_count = Counter(wifi_info)
        wifi_ans = [key for key, value in wifi_count.iteritems() if value >= 20]

        # print "wifi number: " + str(len(wifi_ans))
        # print "sample number: ", len(data)

        # 构建数据的类别
        columns = []
        label = list(data['shop_id'])
        shop = list(set(label))
        y = []
        for shop_id in label:
            index = shop.index(shop_id)
            y.append(index)
        # get long and lati and time feature
        print Counter(y)

        long = list(data['longitude'])
        lati = list(data['latitude'])
        times = list(data['time_stamp'])
        timess = []

        for array in times:
            ay = array.split()
            t = ay[1].split(':')
            t = int(t[0]) * 60 + int(t[1])
            timess.append(t)

        # 统计连接的wifi特征,统计连接次数大于1的wifi连接构建连接特征
        df1 = df[df.connect == "true"]
        wifi_true = list(set(list(df1['bssid'])))
        wifi_fliter = []

        data_true = pd.DataFrame()
        data_true['label'] = y

        for wifi in wifi_true:
            data = df1[df1.bssid == wifi]
            if len(data) >= 2:
                wifi_fliter.append(wifi)

        wifi_connect = ["true-" + i for i in wifi_fliter]
        for i in range(len(wifi_fliter)):
            data_true[wifi_connect[i]] = 0
        data_true = data_true[wifi_connect]

        for index, value in df1.iterrows():
            bssid = value[0]
            row = int(value[2])
            if bssid in wifi_fliter:
                index_c = wifi_fliter.index(bssid)
                data_true.iat[row, index_c] = 1

        # 内存回收
        del data
        gc.collect()
        del df1
        # del data_true
        gc.collect()
        print data_true.shape
        # columns = columns + wifi_connect

        # 去掉店铺中只出现一次的wifi，对剩下的wifi取交集
        wifi_tmp = []
        for shop_id in shop:
            for index, value in shop_bssid[shop_id].iteritems():
                if value >= 3 and index not in wifi_tmp and index in wifi_ans:
                    wifi_tmp.append(index)

        # 对于一个wifi对应多个店铺的情况，去掉这些wifi---------对准确率没有改善
        # wifi_less_shop = []
        # for bssid in wifi_tmp:
        #     shop_dis = wifi_shop_num[bssid]
        #     shop_number = len(shop_dis)
        #     if shop_number<35:
        #         wifi_less_shop.append(bssid)
        # wifi_ans = wifi_less_shop
        wifi_ans = list(set(wifi_tmp + wifi_fliter))

        # 对wifi建立索引
        wifi_index = defaultdict(lambda: 0)
        for index, value in enumerate(wifi_ans):
            wifi_index[value] = index
        # print "1"
        # 构建特征数据
        data_tmp = pd.DataFrame()
        data_tmp['label'] = y
        df = df[df['bssid'].isin(wifi_ans)]
        for wifi in wifi_ans:
            data_tmp[wifi] = -80
        data_tmp = data_tmp[wifi_ans]
        # data_tmp['long'] = long
        # data_tmp['lati'] = lati
        # data_tmp['time'] = timess
        print data_tmp.shape
        print len(y)

        for index, data in df.iterrows():
            bssid = data[0]
            row = int(data[2])
            strength = max(float(data[3]), -75)
            strength = (strength + 75) / 55
            index_c = wifi_index[bssid]
            data_tmp.iat[row, index_c] = strength

        # data_tmp = pd.concat([data_tmp,data_true],axis = 1)
        # print data_tmp.shape
        # print "2"
        # 对特征进行离散化处理
        # data_tmp[data_tmp<=-80] = 1
        # data_tmp[(data_tmp>-80)&(data_tmp<=-70)] =2
        # data_tmp[(data_tmp>-80)&(data_tmp<=-60)] = 3
        # data_tmp[(data_tmp>-60)&(data_tmp<=-50)] = 4
        # data_tmp[(data_tmp>-50)&(data_tmp<=-40)] = 5
        # data_tmp[(data_tmp>-40)&(data_tmp<=-30)] = 6
        # data_tmp[(data_tmp>-30)&(data_tmp<=-20)] = 7
        # data_tmp[(data_tmp>-20)&(data_tmp<0)] = 8
        # data_tmp.to_csv(r"index.csv",index=None)
        # data_new = pd.DataFrame()
        # print "3"

        # 回收内存
        del data_true
        del df
        gc.collect()

        # wifi_new = []
        # res = 1
        # data_list = {}
        # start = time.time()
        # for wifi in wifi_ans:
        #     s_l = list(data_tmp[wifi])
        #     for i in range(1,7):
        #         column = str(i) + "-" + wifi
        #         series = [1 if x==i else 0 for x in s_l]
        #         data_list[column] = series
        #         # data_new[column] = series
        #         # series = data_key.map(lambda x: 1 if x==i else 0)
        #         # series = pd.DataFrame(series,columns=)
        #         # data_new = pd.concat([data_new, series], axis=1,join_axes=[data_new.index])
        #         # data_true = pd.concat([data_true, series], axis=1, join_axes=[data_true.index])
        #         wifi_new.append(column)
        #     res += 1

        # end = time.time()
        # print end-start
        # data_new = pd.DataFrame.from_dict(data_list)
        # wifi_ans = wifi_new
        # print "done"

        # del data_list
        # del data_tmp
        # gc.collect()
        # end = time.time()
        # print "data time: ", end-start, "shape: ", data_new.shape

        # columns = wifi_ans + columns
        # data_tmp['null'] = (data_tmp > 0.0).sum(axis=1)
        # wifi_ans.append('null')
        # min_max_scler = MinMaxScaler()
        # data = min_max_scler.fit_transform(data_tmp)
        # data = pd.DataFrame(data, columns=wifi_ans)
        # data = pd.concat([data, data_true], axis=1, join_axes=[data.index])

        # scaler = preprocessing.StandardScaler().fit(data_tmp)
        # data = pd.DataFrame(scaler.transform(data_tmp), columns=wifi_ans)

        # train_x, test_x, train_y, test_y = train_test_split(data_tmp.values, y, test_size=0.001, random_state=1)
        # # # #
        # del data_tmp
        # gc.collect()

        # # 对训练集进行归一化
        # scaler = preprocessing.MinMaxScaler().fit(train_x)
        # data_train = pd.DataFrame(scaler.transform(train_x))
        # data_test = pd.DataFrame(scaler.transform(test_x))

        # pca进行降维
        # start = time.time()
        # # pca = IncrementalPCA(n_components=0.95, batch_size=100*(train_x.shape[1]))
        # pca = PCA(n_components=0.96)
        # pca.fit(train_x)
        # x_train = pca.transform(train_x)
        # x_test = pca.transform(test_x)
        # end = time.time()
        #
        # print "pca time: ", end-start, "shape: ", x_train.shape
        # x_train = pca.transform(train_x)
        # x_test = pca.transform(test_x)
        # x_train = train_x
        # x_test = test_x

        # del train_x
        # del test_x
        # gc.collect()

        start = time.time()
        # 使用分类器训练和预测模型
        # clf = LogisticRegression(C=1.0, multi_class='multinomial',
        #                          penalty='l2', solver='sag', tol=0.01)
        # clf.fit(x_train, train_y)
        # best  params
        # {'n_neighbors': 21}
        # best_score
        # 0.514038110683

        tuned_parameters = [{'n_neighbors': range(21,30,2)}]
        gsearch1 = GridSearchCV(KNeighborsClassifier(algorithm = 'auto',n_neighbors=21),
                                tuned_parameters, cv=5)
        # gsearch1.fit(data_tmp.values, y)
        # clf = KNeighborsClassifier(algorithm = 'auto',)

        # gsearch1 = KNeighborsClassifier(n_neighbors = 101,)
        # gsearch1.fit(data_tmp.values, y)
        # scores = cross_val_score(gsearch1, data_tmp.values, y,cv=5 )
        # print scores
            # gsearch1.fit(data_tmp, y)
            # print i
        print 'best params', gsearch1.best_params_
        print 'best_score', gsearch1.best_score_

        # print gsearch1.score(data_train,train_y)
        # print gsearch1.score(data_test,test_y)

        end = time.time()
        print "train time: ", end - start
        # start = time.time()
        # # y_pre = clf.predict_proba(x_test)
        #
        # y_pre = gsearch1.predict(test_x)
        # res = 0
        # for i in range(len(y_pre)):
        #     if y_pre[i] == test_y[i]:
        #         res += 1
        # scores = res / float(len(y_pre))

        # del clf
        # # del pca
        # del train_x
        # del test_x
        # del train_y
        # gc.collect()
        # end = time.time()
        # print "predict time:", end -start

        return scores

    def main(self):
        """
        主程序
        :return:
        """
        data_behavior = pd.read_csv("m_data.csv")
        df = pd.DataFrame()
        mall = list(set(list(data_behavior['mall_id'])))
        df['mall_id'] = mall
        df['score'] = 0.0
        res = 0
        start = time.time()
        for mall_id in mall:
            print "mall count: ", res + 1
            start1 = time.time()
            if mall_id == "m_7800":
                pass
            else:
                continue
            data_mall = data_behavior[data_behavior.mall_id == mall_id]

            score = self.generate_train_data(data_mall)
            df.loc[res, ['score']] = score
            end1 = time.time()
            t = end1 - start1
            print "mall: ", mall_id, "length: ", len(data_mall), "  score: ", score, "use time: ", t
            del data_mall
            gc.collect()
            res += 1
        # df.to_csv(r"all malls.csv", index=None)
        end = time.time()
        k = (start - end) / 60
        print "use time: ", -k


if __name__ == '__main__':
    f = SHDW()
    f.main()



# -*- coding:utf-8 -*-
__author__ = 'ZLZ'
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA
import time
from sklearn.preprocessing import MinMaxScaler
import gc
from sklearn.ensemble import RandomForestClassifier


class SHDW(object):
    def __init__(self):

        self.C = None

    def generate_train_data(self, data,testdata):
        """
        对wifi信息生成数据特征,筛选部分特征，并对特征做归一化处理
        :return:
        """
        start = time.time()

        # dnf = data.copy()
        # dnf.loc[:, ['time_stamp']] = pd.to_datetime(dnf['time_stamp'])
        # data = dnf.sort_values(by='time_stamp')
        # dsf = testdata.copy()
        # dsf.loc[:, ['time_stamp']] = pd.to_datetime(dsf['time_stamp'])
        # testdata = dsf.sort_values(by='time_stamp')

        # del dnf
        # del dsf
        # gc.collect()


        # testweekdays = []
        # testhours = []
        # for index, row in testdata.iterrows():
        #     wd = row['time_stamp'].weekday()
        #     testweekdays.append(wd)
        #     hr = row['time_stamp'].hour
        #     testhours.append(hr)
        #
        #
        # weekdays = []
        # hours = []
        # for index, row in data.iterrows():
        #     wd = row['time_stamp'].weekday()
        #     weekdays.append(wd)
        #     hr = row['time_stamp'].hour
        #     hours.append(hr)

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


        testwifiDict = {
            'index': [],
            'bssid': [],
            'strength': [],
            'connect': []
        }
        res = 0
        for index, row in testdata.iterrows():
            for wifi in row.wifi_infos.split(';'):
                info = wifi.split('|')
                testwifiDict['index'].append(res)
                testwifiDict['bssid'].append(info[0])
                testwifiDict['connect'].append(info[2])
                testwifiDict['strength'].append(info[1])
            res += 1
        testdf = pd.DataFrame(testwifiDict)



        # 对我出现的次数进行统计,并去掉出现次数少的wifi
        wifi_info = list(df['bssid'])
        wifi_count = Counter(wifi_info)
        wifi_ans = [key for key, value in wifi_count.iteritems() if value >= 3]

        # print "wifi number: " + str(len(wifi_ans))
        # print "sample number: ", len(data)

        # 构建数据的类别

        label = list(data['shop_id'])
        shop = list(set(label))
        y = []
        for shop_id in label:
            index = shop.index(shop_id)
            y.append(index)
        # get long and lati and time feature

        long = list(data['longitude'])
        lati = list(data['latitude'])
        times = list(data['time_stamp'])
        timess = []

        for array in times:
            ay = array.split()
            t = ay[1].split(':')
            t = int(t[0]) * 60 + int(t[1])
            timess.append(t)

        testtimes = list(testdata['time_stamp'])
        testtimess = []

        for array in testtimes:
            ay = array.split()
            t = ay[1].split(':')
            t = int(t[0]) * 60 + int(t[1])
            testtimess.append(t)

        # 统计连接的wifi特征,统计连接次数大于1的wifi连接构建连接特征
        df1 = df[df.connect == "true"]
        wifi_true = list(set(list(df1['bssid'])))
        wifi_fliter = []

        data_true = pd.DataFrame()
        data_true['label'] = y

        for wifi in wifi_true:
            data = df1[df1.bssid == wifi]
            if len(data) >= 3:
                wifi_fliter.append(wifi)
        # print len(wifi_fliter)
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
        del data_true
        gc.collect()

        # columns = columns + wifi_connect

        # 去掉店铺中只出现一次的wifi，对剩下的wifi取交集
        wifi_tmp = []
        for shop_id in shop:
            for index, value in shop_bssid[shop_id].iteritems():
                if value > 1 and index not in wifi_tmp and index in wifi_ans:
                    wifi_tmp.append(index)


        wifi_ans =['long','lati']+wifi_tmp

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
            data_tmp[wifi] = -90
        data_tmp = data_tmp[wifi_ans]
        data_tmp['long'] = long
        data_tmp['lati'] = lati
        # data_tmp['weekday'] = weekdays
        data_tmp['time'] = timess


        for index, data in df.iterrows():
            bssid = data[0]
            row = int(data[2])
            strength = int(data[3])
            index_c = wifi_index[bssid]
            data_tmp.iat[row, index_c] = strength
        # 回收内存


        data_tmp_1 = pd.DataFrame()
        data_tmp_1['row_id'] = testdata['row_id']
        for wifi in wifi_ans:
            data_tmp_1[wifi] = -90
        data_tmp_1 =  data_tmp_1[wifi_ans]
        data_tmp_1['long'] = list(testdata['longitude'])
        data_tmp_1['lati'] = list(testdata['latitude'])
        # data_tmp_1['weekday'] = testweekdays
        data_tmp_1['time'] = testtimess

        for index, data in testdf.iterrows():
            bssid = data[0]
            row = int(data[2])
            strength = int(data[3])
            index_c = wifi_index[bssid]
            data_tmp_1.iat[row, index_c] = strength


        del testdf
        gc.collect()

        # # 对训练集进行归一化
        # scaler = preprocessing.StandardScaler().fit(train_x)
        # data_train = pd.DataFrame(scaler.transform(train_x), columns=wifi_ans)
        # data_test = pd.DataFrame(scaler.transform(test_x), columns=wifi_ans)

        start = time.time()
        clf = RandomForestClassifier(n_estimators=190, min_samples_leaf=2, max_features='sqrt', criterion="gini",
                                     min_samples_split=6, oob_score=True, class_weight="balanced")
        clf.fit(data_tmp, y)
        y_pre = clf.predict(data_tmp_1)

        result = pd.DataFrame()
        result['row_id'] = testdata['row_id']
        result['shop_id'] = '0'
        for j in range(len(y_pre)):
            result.iloc[j, 1] = shop[y_pre[j]]

        end = time.time()
        print "train time: ", end - start

        del clf
        del data_tmp
        del testdata
        gc.collect()
        return result

    def main(self):
        """
        主程序
        :return:
        """
        data_behavior = pd.read_csv("behavior data version2.csv")
        mall = list(set(list(data_behavior['mall_id'])))
        evaluation = pd.read_csv(u'AB榜测试集-evaluation_public.csv')
        res = 0
        start = time.time()
        result = pd.DataFrame()
        for mall_id in mall:
            print "mall count: ", res + 1
            start1 = time.time()
            # if mall_id == "m_7800":
            #     pass
            # else:
            #     continue
            data_test = evaluation[evaluation.mall_id == mall_id]
            data_mall = data_behavior[data_behavior.mall_id == mall_id]
            score = self.generate_train_data(data_mall, data_test)
            result = pd.concat([result,score],axis = 0)
            # df.loc[res, ['score']] = score
            end1 = time.time()
            t = end1 - start1
            print "mall: ", mall_id, "length: ", len(data_mall),  "use time: ", t
            del data_mall
            del data_test
            gc.collect()
            res += 1
        result = result.sort_values(by='row_id')
        result.to_csv(r"result.csv", index=None)
        end = time.time()
        k = (start - end) / 60
        print "use time: ", -k


if __name__ == '__main__':
    f = SHDW()
    f.main()



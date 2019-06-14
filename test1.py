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
        wifi_ans = [key for key, value in wifi_count.iteritems() if value >= 3]

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
        # print data_true.shape
        # columns = columns + wifi_connect

        # 去掉店铺中只出现一次的wifi，对剩下的wifi取交集
        wifi_tmp = []
        for shop_id in shop:
            for index, value in shop_bssid[shop_id].iteritems():
                if value >= 2 and index not in wifi_tmp and index in wifi_ans:
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
            data_tmp[wifi] = -90
        data_tmp = data_tmp[wifi_ans]
        # data_tmp['long'] = long
        # data_tmp['lati'] = lati
        # data_tmp['time'] = timess
        # print data_tmp.shape

        for index, data in df.iterrows():
            bssid = data[0]
            row = int(data[2])
            strength = max(float(data[3]), -90)
            index_c = wifi_index[bssid]
            data_tmp.iat[row, index_c] = strength

        data_tmp = pd.concat([data_tmp,data_true],axis = 1)

        # 回收内存
        del data_true
        del df
        gc.collect()



        train_x, test_x, train_y, test_y = train_test_split(data_tmp.values, y, test_size=0.2, random_state=3)
        # # #
        print len(test_y)
        print len(train_y)

        del data_tmp
        gc.collect()


        start = time.time()


        clf = RandomForestClassifier(n_estimators=190, min_samples_leaf=1, max_features='sqrt', criterion="gini",
                                     min_samples_split=6, oob_score=True, class_weight="balanced")

        clf.fit(train_x, train_y)

        print clf.score(train_x,train_y)
        print clf.score(test_x,test_y)


        end = time.time()
        print "train time: ", end - start
        start = time.time()

        y_pre = clf.predict(test_x)
        res = 0

        y_3 = []

        # print len(test_y),len(y_pre)


        for i in range(len(y_pre)):

            if y_pre[i] == test_y[i]:
                res += 1
            # else:
            #     y_2.append(y_pre[i])
            #     y_erro.append(test_y[i])
        score = res / float(len(y_pre))
        print score




        del clf
        del test_y
        del train_x
        del test_x
        del train_y
        gc.collect()
        end = time.time()
        print "predict time:", end - start
        return score

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
        df.to_csv(r"all malls.csv", index=None)
        end = time.time()
        k = (start - end) / 60
        print "use time: ", -k


if __name__ == '__main__':
    f = SHDW()
    f.main()


# 52: 12, 9: 9, 43: 8, 44: 8, 21: 6, 50: 6, 63: 6, 6: 5, 11: 5, 28: 5, 47: 5, 38: 4, 45: 4, 33: 3, 60: 3, 65: 3, 29: 2, 71: 2, 0: 1, 1: 1, 39: 1, 42: 1, 46: 1, 67: 1, 74: 1, 86: 1, 87: 1})
# Counter({9: 49, 52: 22, 29: 18, 21: 17, 50: 17, 71: 12, 44: 11, 46: 11, 45: 10, 47: 10, 1: 9, 11: 9, 28: 9, 33: 9, 38: 9, 65: 9, 6: 8, 43: 8, 63: 8, 42: 7, 78: 7, 87: 7, 15: 6, 75: 6, 88: 6, 40: 5, 49: 5, 58: 5, 60: 5, 68: 5, 61: 4, 73: 4, 4: 3, 10: 3, 26: 3, 35: 3, 51: 3, 55: 3, 72: 3, 74: 3, 7: 2, 8: 2, 12: 2, 14: 2, 16: 2, 20: 2, 32: 2, 56: 2, 59: 2, 69: 2, 77: 2, 80: 2, 85: 2, 0: 1, 5: 1, 19: 1, 22: 1, 34: 1, 36: 1, 37: 1, 39: 1, 41: 1, 48: 1, 57: 1, 64: 1, 67: 1, 76: 1, 79: 1, 81: 1, 84: 1, 86: 1})
# Counter({9:
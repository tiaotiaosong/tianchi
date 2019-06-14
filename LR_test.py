# -*- coding:utf-8 -*-
__author__ = 'ZLZ'
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import time
from sklearn.decomposition import PCA
import gc


class SHDW(object):
    def __init__(self):
        
        self.C = None
    
    def generate_train_data(self, data):
        """
        对wifi信息生成数据特征,筛选部分特征，并对特征做归一化处理
        :return:
        """
        wifiDict = {
            'index': [],
            'bssid': [],
            'strength': [],
            'connect': []
        }
        res = 0
        for index, row in data.iterrows():
            for wifi in row.wifi_infos.split(';'):
                info = wifi.split('|')
                wifiDict['index'].append(res)
                wifiDict['bssid'].append(info[0])
                wifiDict['strength'].append(info[1])
                wifiDict['connect'].append(info[2])
            res += 1
        df = pd.DataFrame(wifiDict)
        
        # 对我出现的次数进行统计,并去掉出现次数少的wifi
        wifi_info = list(df['bssid'])
        wifi_count = Counter(wifi_info)
        wifi_ans = [key for key, value in wifi_count.iteritems() if value >= 0]
        df = df[df['bssid'].isin(wifi_ans)]
        print "wifi number: " + str(len(wifi_ans))
        print "sample number: ", len(data)
        
        # 构建数据的类别
        label = list(data['shop_id'])
        y = []
        shop = list(set(label))
        for shop_id in label:
            index = shop.index(shop_id)
            y.append(index)
        
        # 对wifi建立索引
        wifi_index = defaultdict(lambda: 0)
        for index, value in enumerate(wifi_ans):
            wifi_index[value] = index
        
        # 构建特征数据
        data_tmp = pd.DataFrame()
        data_tmp['label'] = y
        for wifi in wifi_ans:
            data_tmp[wifi] = -150
        data_tmp = data_tmp[wifi_ans]
        for index, data in df.iterrows():
            bssid = data[0]
            row = int(data[2])
            strength = int(data[3])
            index_c = wifi_index[bssid]
            data_tmp.iat[row, index_c] = strength
        
        train_x, test_x, train_y, test_y = train_test_split(data_tmp, y, test_size=0.2)
        del data_tmp
        del df
        
        # 对训练集进行归一化
        scaler = preprocessing.StandardScaler().fit(train_x)
        data_train = pd.DataFrame(scaler.transform(train_x), columns=wifi_ans)
        data_test = pd.DataFrame(scaler.transform(test_x), columns=wifi_ans)
        
        # pca进行降维
        # pca = PCA(n_components=0.95)
        # pca.fit(data_train)
        # x_train = pca.transform(data_train)
        # x_test = pca.transform(data_test)
        
        # 使用lda进行降维
        # lda = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
        #                                  solver='svd', store_covariance=False, tol=0.0001)
        # lda.fit(data_train, train_y)
        # x_train = lda.transform(data_train)
        # x_test = lda.transform(data_test)
        
        # 使用分类器训练和预测模型
        clf = LogisticRegression(C=10.0 / len(data_train), multi_class='ovr',max_iter=300,
                                 penalty='l1', solver='liblinear', tol=0.00001)
        
        clf.fit(train_x, train_y)
        y_pre = clf.predict(test_x)
        res = 0
        for i in range(len(y_pre)):
            shop_c = test_y[i]
            shop_p = shop[y_pre[i]]
            if y_pre[i] == test_y[i]:
                res += 1
        score = res / float(len(y_pre))
        
        return score
    
    def main(self):
        """
        主程序
        :return:
        """
        data_behavior = pd.read_csv("behavior data version2.csv")
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
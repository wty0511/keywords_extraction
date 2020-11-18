import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np
import matplotlib.pylab as plt
import lightgbm as gbm
from matplotlib.font_manager import FontProperties
from sklearn.metrics import classification_report as report


class train():

    def __init__(self):

        self.acc = 0
        print("开始训练")

    def training(self):
        data = pd.read_csv("./test.csv")
        acc=0
        #cols = [col for col in data.columns if col not in ['id', '关键词', '标签']]
        # cols = [col for col in data.columns if col  in['头词频', '词长', '首次出现词位置', '最后出现词位置', '位置跨度', '词平均', '词偏度', '首次句位置', '最大句差', '度中心性', '接近中心性', '介数中心性', '特征向量中心性', '最小句中位置', '平均句中位置', 't', 'v', 'b', 'z', 'r', 'm', 'd', 'u', 'e', 'x']]
        cols = [col for col in data.columns if col in ['词频', '首次出现词位置']]
        x = data.loc[:, cols]
        y = data.loc[:, '标签']
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        model=lgb.Booster(model_file='lgbmodel_allfeature.model')
        y_pred = model.predict(x)
        y_pred = list(y_pred)
        Y_val = list(y)
        i = 0
        count = 0

        for item in y_pred:
            if item > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

            if y_pred[i] == Y_val[i]:
                count = count + 1
            i = i + 1
        print(report(Y_val, y_pred))
        acc = acc + count / i

        print(acc)



train().training()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
import numpy as np
import matplotlib.pylab as plt

from matplotlib.font_manager import FontProperties
'''
class train():

    def __init__(self,decode):
        self.decode=decode
        self.acc=0
        print("开始训练")
    def training(self):
        data = pd.read_csv("./balance_std_new.csv")
        #print(data)
        col = [col for col in data.columns if col  not in ['id','关键词','标签']]
        cols=[]
        for i,c in zip(self.decode,col):
            if i==1:
                cols.append(c)
        x = data.loc[:,cols ]
        y = data.loc[:, '标签']
        x=x.reset_index(drop=True)
        y=y.reset_index(drop=True)

        #print(col)
        #print(self.decode)
        print(cols)

        # 测试集为30%，训练集为70%
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc=0
        for i, (train_index, val_index) in enumerate(kf.split(x_train)):

            X_train  = x_train.iloc[train_index]

            Y_train = y_train.iloc[train_index]

            X_val = x_train.iloc[val_index]

            Y_val = y_train.iloc[val_index]


            lgb_train = lgb.Dataset(
                X_train, Y_train)

            lgb_eval = lgb.Dataset(
                X_val, Y_val,
                reference=lgb_train)
            #     print('开始训练......')
            params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': {'auc'},
                    'learning_rate': 0.025,
                    'num_leaves': 100,
                    'min_data_in_leaf': 50,
                    'bagging_fraction': 0.85,
                    'seed':42
            }
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=10000,
                            valid_sets=lgb_eval,
                            early_stopping_rounds=10,
                            verbose_eval=False,

                            )

            y_pred = gbm.predict(x_test)

            y_pred= list(y_pred)
            Y_val=list(y_test)
            i = 0
            count = 0

            for item in y_pred:
                if item>0.5:
                    y_pred[i]=1
                else:
                    y_pred[i] = 0

                if y_pred[i]==Y_val[i]:
                    count=count+1
                i=i+1
            acc= acc+count/i
        print(acc/5)
        self.acc=acc/5
'''

data_train = pd.read_csv("./train_balance.csv")
data_test = pd.read_csv("./test_balance.csv")
class train():

    def __init__(self,decode):
        self.decode = decode
        self.acc = 0

        print("开始训练")

    def training(self):

        col = [col for col in data_train.columns if col not in ['id', '关键词', '标签']]
        cols = []
        for i, c in zip(self.decode, col):
            if i == 1:
                cols.append(c)



        x_train = data_train.loc[:, cols]
        y_train = data_train.loc[:, '标签']
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        x_val = data_test.loc[:, cols]
        y_val = data_test.loc[:, '标签']
        x_val = x_val.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        # 测试集为30%，训练集为70%
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)





        lgb_train = lgb.Dataset(
            x_train, y_train)

        lgb_eval = lgb.Dataset(
            x_val, y_val,
            reference=lgb_train)
        #     print('开始训练......')

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'learning_rate': 0.025,
            'num_leaves': 100,

            'min_data_in_leaf': 70,
            'bagging_fraction': 0.85,
            'is_unbalance': 'true',
            'seed': 42
        }


        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=5000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=30,
                        verbose_eval=False,

                        )

        y_pred = gbm.predict(x_val)
        y_pred = list(y_pred)
        Y_val = list(y_val)
        pos=0
        pos_acc=0
        pos_pre=0
        for i, j in zip(Y_val, y_pred):
            if (i >=0.5):
                pos += 1

            if (i >= 0.5 and j>=0.5):
                pos_acc += 1
            if (j >= 0.5):
                pos_pre += 1

        pos_r = pos_acc / pos
        pos_a = pos_acc / pos_pre
        self.acc=(pos_a*pos_r)/(pos_a+pos_r)*2




        #gbm.save_model('lgbmodel_allfeature.model')
'''
    print(pd.DataFrame({
            'column': cols,
            'importance': gbm.feature_importance(),
        }).sort_values(by='importance'))

     column  importance
20  出现在最后一句           0
3     出现在标题           0
19   出现在第一句          28
30     包含英文          47
29     最小句差          48
31     包含数字          53
28     最大句差          89
6      位置跨度          92
7       词方差         103
18    最后句位置         110
22      句方差         114
27     句差平均         155
24      句偏度         171
10      词峰度         174
9       词偏度         193
8       词平均         194
5   最后出现词位置         196
17    首次句位置         203
13     最大词差         205
11     词差方差         214
23      句平均         216
25      句峰度         262
26     句差方差         267
33     度中心性         296
21   句子出现频率         311
16     最短句长         336
12     词差平均         345
14     最小词差         392
34    接近中心性         431
0       头词频         447
2        词长         448
37   共现矩阵偏度         470
15     最长句长         472
36  特征向量中心性         495
1        词频         641
35    介数中心性         662
32      相似度         680
4   首次出现词位置         736
['首次出现词位置','相似度','介数中心性','词频','特征向量中心性','最长句长','共现矩阵偏度','词长','头词频','接近中心性','最小词差','词差平均','最短句长','度中心性','句差方差','句峰度','句平均','词差方差','最大词差','','']
'''

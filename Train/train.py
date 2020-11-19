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

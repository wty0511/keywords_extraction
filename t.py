import pandas as pd

import lightgbm as lgb

from sklearn.metrics import classification_report as report

class train():

    def __init__(self):

        self.acc = 0
        print("开始训练")

    def training(self):
        data_train = pd.read_csv("./train_balance.csv")
        data_test = pd.read_csv("./test_balance.csv")

        acc=0
        #cols = [col for col in data_train.columns if col not in ['id', '关键词', '标签']]
        #cols = [col for col in data_train.columns if col  in ['头词频','词频','词长','IDF','出现在标题','首次出现词位置','最后出现词位置','词方差','词平均','词偏度','词峰度','词差方差','最大词差','最小词差','最小句中位置','首次句位置','最后句位置','出现在第一句','出现在最后一句','句子出现频率','句平均','句偏度','包含英文','度中心性','接近中心性','s','f','v','d','k','x','i','l','un','包含数字']]
        '''
        cols=['词频','词长','IDF','出现在标题','首次出现词位置','最后出现词位置','词方差','词偏度','最大句中位置','最小句中位置',
              '平均句中位置','平均句长','首次句位置','出现在最后一句','句子出现频率','句方差',
              '句平均','句差方差','最大句差','包含英文','接近中心性','n', 't', 'v', 'z', 'q', 'd', 'k', 'x', 'y', '包含数字']
        
         ['词频', '词长', 'IDF', '出现在标题', '首次出现词位置', '词方差', '词平均', '最大词差', '最大句中位置', '平均句中位置', 
         '首次句位置', '出现在第一句', '出现在最后一句', '句子出现频率', '句方差', '句差方差', '最大句差', '度中心性',
          'n', 'v', 'a', 'z', 'd', 'h', 'k', 'x', 'g', 'j', 'y', 'un', '包含数字']

         '''

        #cols=['头词频', '词频', '词长', 'TFIDF', 'IDF', '首次出现词位置', '最后出现词位置', '词方差', '词平均', '词峰度', '词差方差', '词差平均', '最大句中位置', '首次句位置', '句偏度', '句峰度', '接近中心性', '介数中心性', 'n', 'v',  'x']
        cols=['TFIDF','首次出现词位置']
        #cols = ['头词频', '词频', '词长', 'TFIDF', 'IDF', '首次出现词位置', '最后出现词位置', '词方差', '词平均', '词峰度', '词差方差', '词差平均', '最大句中位置', '首次句位置', '句偏度', '句峰度', '包含英文', '接近中心性', '介数中心性', 'n', 'v', 'b', 'z', 'h', 'k', 'x', 'g']
        #cols=['词频','词长','TFIDF','IDF','首次出现词位置','平均句中位置','出现在标题','平均句长','最后出现词位置','位置跨度','最大句中位置','最小句中位置','最长句长','最短句长']
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

        i=0
        for item in y_pred:
            if item > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

            i = i + 1
        print(report(Y_val, y_pred,digits=4))
        gbm.save_model('lgbmodel_youhua.model')

train().training()
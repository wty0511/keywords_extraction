import pandas as pd
import jieba
import math
import json
import numpy as np

data =pd.read_csv('./new_token.csv')
c=[]
jieba.load_userdict('./字典/明星.txt')
jieba.load_userdict('./字典/实体名词.txt')
jieba.load_userdict('./字典/歌手.txt')
jieba.load_userdict('./字典/动漫.txt')
jieba.load_userdict('./字典/电影.txt')
jieba.load_userdict('./字典/电视剧.txt')
jieba.load_userdict('./字典/流行歌.txt')
jieba.load_userdict('./字典/创造101.txt')
jieba.load_userdict('./字典/百度明星.txt')
jieba.load_userdict('./字典/美食.txt')
jieba.load_userdict('./字典/FIFA.txt')
jieba.load_userdict('./字典/NBA.txt')
jieba.load_userdict('./字典/网络流行新词.txt')
jieba.load_userdict('./字典/显卡.txt')

## 爬取漫漫看网站和百度热点上面的词条
jieba.load_userdict('./字典/漫漫看_明星.txt')
jieba.load_userdict('./字典/百度热点人物+手机+软件.txt')
jieba.load_userdict('./字典/自定义词典.txt')

## 实体名词抽取之后的结果 有一定的人工过滤
## origin_zimu 这个只是把英文的组织名过滤出来
jieba.load_userdict('./字典/person.txt')
jieba.load_userdict('./字典/origin_zimu.txt')

## 第一个是所有《》里面出现的实体名词
## 后者是本地测试集的关键词加上了
jieba.load_userdict('./字典/出现的作品名字.txt')
jieba.load_userdict('./字典/val_keywords.txt')
l=[]
for tup in data.itertuples():

    l.extend(tup[3].split(' '))
    l.extend(tup[4].split(' '))
#print(l)
l=list(set(l))
dic={}
file = open('./idf.txt','w',encoding='utf-8')
for word in l:
    c=0
    for tup in data.itertuples():
        article=tup[3].replace(' ','')+tup[4].replace(' ','')
        if word in article:
            c+=1
    file.write(word+' '+str(math.log(999/(c+1),10))+'\n')



#json.dump(dic, file,ensure_ascii=False,indent=2)
file.close()

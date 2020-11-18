import gensim
import pandas as pd
from gensim.models import word2vec

import pandas as pd
import jieba



import  re

all_docs_final = pd.read_csv('Train_DataSet.csv', sep=',')

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
jieba.load_userdict('./字典/漫漫看_明星.txt')
jieba.load_userdict('./字典/百度热点人物+手机+软件.txt')
jieba.load_userdict('./字典/自定义词典.txt')
jieba.load_userdict('./字典/person.txt')
jieba.load_userdict('./字典/origin_zimu.txt')
jieba.load_userdict('./字典/出现的作品名字.txt')
jieba.load_userdict('./字典/val_keywords.txt')

i=0
file=pd.DataFrame()
file["id"]=None
file["title"]=None
file["context"]=None
file["keywords"]=None


for index, row in all_docs_final.iterrows():
    outstr_1 = ""
    outstr_2 = ""
    stopwords = [line.strip() for line in open('./stopword.txt', encoding='UTF-8').readlines()]
    word_list_1 = jieba.cut(str(row[1]), cut_all=False)
    sentence_list=scen_list = list(filter(None, re.split('[。？！；!?]', str(row[2]).strip())))
    word_list_2=[]
    for sentence in sentence_list:
        print(sentence)
        word_list_2.extend(list(jieba.cut(sentence, cut_all=False)))
        word_list_2.append('')


    a = list(word_list_1)
    b = list(word_list_2)
    i=i+1
    print(word_list_2)
    for word in a:

        if word not in stopwords:

            if word != '\n':
                outstr_1 += word
                outstr_1 += " "


    for word in b:

        if word not in stopwords:

            if word != '\n':
                outstr_2 += word
                outstr_2 += " "

    new = pd.DataFrame({'id': row[0],
                        'title': outstr_1,
                        'context': outstr_2,
                        'keywords':row[3]
                        },
                       index=[1])
    file=file.append(new,ignore_index=True)

    print(i)


file.to_csv('new_token.csv',header=1,index=1)

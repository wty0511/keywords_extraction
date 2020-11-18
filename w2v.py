import gensim
import pandas as pd
from gensim.models import word2vec

import pandas as pd
import jieba
all_docs_final = pd.read_csv('all_docs_final.csv', sep=',')
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
outstr = ""
stopwords = [line.strip() for line in open('./stopword.txt', encoding='UTF-8').readlines()]
for index, row in all_docs_final.iterrows():

    word_list = jieba.cut(str(row[1]) + str(row[2]), cut_all=False)
    b = list(word_list)

    for word in b:

        if word not in stopwords:

            if word != '\n':
                outstr += word
                outstr += " "
with open('./corpus1.txt', 'w+', encoding='utf-8') as f:
    f.write(outstr)  # 读取的方式和写入的方式要一致
sentences = word2vec.Text8Corpus(r'./corpus1.txt' )
model = word2vec.Word2Vec(sentences, size=100, sg=1,window=5, min_count=1, workers=10)
model.save('./w2v/word2vec.model')

'''
model = gensim.models.Word2Vec.load('./w2v/word2vec.model')
count=0
file=pd.read_table("./train_docs_keywords.txt")
for index, row in file.iterrows():
    list=row[1].split(",")
    for word in list:
        try:
            model.most_similar(word, topn=1)
        except BaseException:
            count=count+1
print(count)
'''
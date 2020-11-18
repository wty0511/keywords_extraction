import re

import gensim
import Graph.graph as graph
import pandas as pd
import jieba
import jieba.analyse

import numpy as np
import string
from scipy.stats import skew, kurtosis
from gensim.models import Doc2Vec



def is_legal_char(uchar):
    if (uchar >= u'\u4e00' and uchar <= u'\u9fa5') or uchar=='。'or uchar=='？'or uchar=='！'or uchar=='；'or uchar=='!'or uchar=='?'or uchar.isdigit() or uchar.isalpha():
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for str in content:
        if is_legal_char(str):
            content_str = content_str +str
    return content_str



def get_candidate_word (title,content):
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
    stopwords = [line.strip() for line in open('./stopword.txt', encoding='UTF-8').readlines()]
    word_list = jieba.cut(title+content, cut_all=False)

    b=list(word_list)
    outstr=""
    for word in b:

        if word not in stopwords:

            if word != '\n':

                outstr += word
                outstr += " "

    candidate_word=jieba.analyse.extract_tags(outstr,allowPOS=('Ag', 'a', 'ad','an', 'b', 'dg', 'd','g','h','i','j','k','l','m','ng','n','nr','ns','nt','nz','q','s','tg','Train','vg','vn','x','z','un'), topK=20,withWeight=False)
    a=list(candidate_word)






    return candidate_word,b
####################################词频
#头词频(归一化)
def get_head_frequency (word_list,candidate):
    count=0
    for i in range(int(len(word_list)/4)):
        if  word_list[i]== candidate:
            count = count + 1
    result =count/len(word_list)
    #print("头词频："+str(result))
    return result

#词频(归一化)
def get_term_frequency (word_list,candidate):
    count=0
    for word in word_list:
        if  word == candidate:
            count = count + 1;
    result = count/len(word_list)
    #print("词频："+str(result))
    return result
######################################词特征
#词长
def len_tags(candidate):

    result = len(candidate)
    #print("词长："+str(result))
    return result

#出现在标题
def occur_in_title(text,candidate):
    if candidate in text:
        return 1
    else:
        return 0
#首次出现位置（词）(归一化)
def fisrt_word_pos(word_list,candidate):
    i=0
    for word in word_list:
        i=i+1
        if word==candidate:
            break

    result =  i/len(word_list)
    #print("首次词位置："+str(result))
    return result


#平均位置（词）（归一化）
def mean_pos(word_list,candidate):
    i=0
    sum=0
    count=0
    result = None
    for word in word_list:
        i=i+1

        if word==candidate:
            count=count+1
            sum=sum+i
    if count==0:
        result = 0
    else:
        result = sum/count/i
    #print("平均词位置："+str(result))
    return  result

#最后出现位置（词）（归一化）
def last_word_pos(word_list,candidate):
    i=0
    pos=0

    for word in word_list:

        i=i+1
        if word==candidate:
            pos=i



        result = pos/len(word_list)
    #("最后词位置："+str(result))

#位置跨度
def get_max_span(word_list, candidate):
    i = 0

    l=[]
    for word in word_list:
        i = i + 1
        if word == candidate:
            l.append(i)


    result = l[-1]-l[0]
    #print("位置跨度："+str(result))


#词位置
def word_pos(word_list, candidate):
    i = 0


    l=[]
    for word in word_list:
        i = i + 1
        if word == candidate:
            l.append(i)



    return l

###########################################句子位置特征
#首次句子位置（归一化）
def firsr_senctence_pos(text, candidate):
    pos=0
    for scen in text:
        pos=pos+1
        if str.find(scen, candidate) != -1:

                break
    result = pos/len(text)
    #print("首次句位置："+str(result))

#平均句长度
def mean_len_sen(text,candidate):
    count=0
    sum=0
    result=None
    for scen in text:
        s=scen.replace(" ", "")
        if str.find(scen,candidate)!=-1:
            count=count+1
            sum=sum+len(s)

    if count==0:
        result= 0
    else:
        result = sum/count
    #print("平均句子长度:"+str(result))
    return result


#最长句长
def max_senctence_len(text,candidate):
    l=[]
    result=None
    for sen in text:
        s = sen.replace(" ", "")
        if str(sen).find(candidate)!=-1:
            l.append(len(s))
    if len(l)>0:
        result=np.max(l)
    else:
        result=0
    #print("最长句长："+str(result))
    return result

#句中位置
def word_position_in_sentence(text,candidate):
    l=[]

    for sen in text:
        s = sen.split(" ")
        try:
            l.append((s.index(candidate)+1)/len(s))
        except:
            continue

    if len(l)>0:
        return np.min(l),np.mean(l),np.max(l)
    else:
        result=0,0,0
    #print("最短句长："+str(result))
    return result

#最短句长
def min_senctence_len(text,candidate):
    l=[]
    result=None
    for sen in text:
        s = sen.replace(" ", "")
        if str(sen).find(candidate)!=-1:
            l.append(len(s))
    if len(l)>0:
        result=np.min(l)
    else:
        result=0
    #print("最短句长："+str(result))
    return result
#句子位置
def sen_pos(text,candidate):
    i=0
    l=[]
    for sen in text:
        i=i+1
        if sen.find(candidate)!=-1:
            l.append(i)
    return l



#出现在第一句
def occur_in_first_sentence(text,candidate):
    result = None
    if candidate in text[0].split(" "):
        result = 1
    else:
        result = 0

    #print("是否出现在第一句" + str(result))
    return result


#出现在最后一句
def occur_in_last_sentence(text,candidate):
    result = None
    if candidate in text[-1].split(" "):
        result = 1
    else:
        result = 0

    #print("是否出现在最后一句"+str(result))
    return result


#出现的句子频率（归一化）
def num_sen(text,candidate):
    count=0

    for scen in text:

        if scen.find(candidate)!=-1:
            count=count+1;

    result= count/len(text)
    #print(count)

    #print("句子频率"+str(result))
    return result
###########################################






#候选词相似度
def get_mean_sim(candidate_vec, mean_vec):
        x = np.array(candidate_vec)
        if np.linalg.norm(x)==0:
            return 0
        y = np.array(mean_vec)
        inner = np.inner(x, y)
        result = inner / np.linalg.norm(x) / np.linalg.norm(y)
        #print("平均相似度：" + str(result))
        return result


## 包含数字
def has_digit(candidate):
    for ch in candidate:
        if ch.isdigit():
            #print("数字：True")
            return 1
    #print("数字：False")
    return 0

##是否包含字母
def has_eng(candidate):
    for ch in candidate:
        if (ch >= u'\u0041' and ch <= u'\u005a') or (ch >= u'\u0061' and ch <= u'\u007a'):
            return 1
        else:
            return 0
    #print("字母：False")
    return False

def get_cixing(text,candidate):
    dic={
        'n':0,
        't': 0,
        's': 0,
        'f': 0,
        'v': 0,
        'a': 0,
        'b': 0,
        'z': 0,
        'r': 0,
        'm': 0,
        'q': 0,
        'd': 0,
        'p': 0,
        'c': 0,
        'u': 0,
        'e': 0,
        'y': 0,
        'o': 0,
        'h': 0,
        'k': 0,
        'x': 0,
        'w': 0,
    }

    for sentence in text:
        sentence_seged = jieba.posseg.cut(sentence.strip())
        for x in sentence_seged:
            try:
                if(x.word==candidate and dic[x.flag[0]]==0):
                    dic[x.flag[0]]+=1
            except:
                continue


    return dic


def get_feature(file):
    res = pd.DataFrame(columns=('头词频', '词频', '词长','出现在标题','首次出现词位置','最后出现词位置','位置跨度','词方差','词平均','词偏度','词峰度','词差方差','词差平均','最大词差','最小词差'
                                ,'最长句长','最短句长','首次句位置','最后句位置','出现在第一句','出现在最后一句','句子出现频率','句方差','句平均','句偏度','句峰度','句差方差','句差平均','最大句差','最小句差',
                                '包含英文','相似度','度中心性','接近中心性','介数中心性','特征向量中心性','共现矩阵偏度','标签','id','关键词','最小句中位置','平均句中位置','最大句中位置',
                                'n','t','s','f','v','a','b','z','r','m','q','d','p','c','u','e','y','o','h','k','x','w'))
    count=0
    lou=0
    zong=0
    for index, row in file.iterrows():
        count+=1
        if(count==899):
            break
        batch = pd.DataFrame(columns=('头词频', '词频', '词长','出现在标题','首次出现词位置','最后出现词位置','位置跨度','词方差','词平均','词偏度','词峰度','词差方差','词差平均','最大词差','最小词差'
                                ,'最长句长','最短句长','首次句位置','最后句位置','出现在第一句','出现在最后一句','句子出现频率','句方差','句平均','句偏度','句峰度','句差方差','句差平均','最大句差','最小句差',
                                '包含英文','相似度','度中心性','接近中心性','介数中心性','特征向量中心性','共现矩阵偏度','标签','id','关键词','最小句中位置','平均句中位置','最大句中位置',
                                'n','t','s','f','v','a','b','z','r','m','q','d','p','c','u','e','y','o','h','k','x','w'))
        #print(batch)
        candidate_list,word_list=get_candidate_word(str(row[2]),str(row[3]))
        #print(row[2])
        #关键词集
        keyword_list=list(row[4].split(","))
        #句子集
        sen_list = list(filter(None,str(row[3]).split('')))
        #print(type(row[2]))
        #总词集
        word_list = list(filter(None,(row[2].split(" ")+re.split('[ ]',(row[3])))))
        #print(word_list)
        #候选词集
        candidate_list=list(set(keyword_list + candidate_list))
        lou+=len(candidate_list)-20
        zong+=len(keyword_list)
        #词图关系
        g=graph.graph(word_list)
        g.build_graph()
        #
        degree_centrality,closeness_centrality,betweenness_centrality,eigenvector_centrality=g.centrality()
        model = gensim.models.Word2Vec.load('./w2v/word2vec.model')
        vec=np.array(100,dtype=float)
        num_candidate_list=len(candidate_list)
        #print(candidate_list)
        co_occurence_metrix = np.zeros((num_candidate_list,num_candidate_list))
        for i in range(num_candidate_list):
            for j in range(i + 1, num_candidate_list):
                count = 0
                for sen in sen_list:
                    if candidate_list[i] in sen and candidate_list[j] in sen:
                        count += 1
                co_occurence_metrix[i, j] = count
                co_occurence_metrix[j, i] = count


        for word in candidate_list:
            try:
                vec=vec+np.array(model[word])
            except BaseException:
                continue

        i=0
        #print(candidate_list)
        for candidate in candidate_list:


            if candidate not in word_list:

                continue



            if candidate in keyword_list:
                tag=1
            else:
                tag=0
            #print(tag)
            #print(candidate)
            dic=get_cixing(sen_list, candidate)
            _word_pos=word_pos(word_list,candidate)
            _sen_pos = sen_pos(sen_list,candidate)


            # 头词频(归一化)
            hf=get_head_frequency(word_list, candidate)

            # 词频(归一化)
            tf=get_term_frequency(word_list, candidate)

            # 词长
            lt=len_tags(candidate)

            # 出现在标题
            oit = occur_in_title(sen_list, candidate)

            # 首次出现位置（词）
            fwp = _word_pos[0]/len(word_list)

            # 最后出现位置（词）
            lwp= _word_pos[-1]/len(word_list)
            w_min_s,w_mean_s,w_max_s, =word_position_in_sentence(sen_list,candidate)
            # 位置跨度
            maxspan= (_word_pos[-1]-_word_pos[0])/len(word_list)

            var_word = np.var(_word_pos)
            mean_word = np.mean(_word_pos)
            skew_word = skew(_word_pos)
            kurt_word = kurtosis(_word_pos)
            #print("diff_word")
            if len(_word_pos) >= 2:
                diff_var_word = np.var(np.diff(_word_pos))
                diff_mean_word = np.mean(np.diff(_word_pos))
                diff_max_word = np.max(np.diff(_word_pos))
                diff_min_word = np.min(np.diff(_word_pos))

            else:
                diff_var_word = 0
                diff_mean_word = 0
                diff_max_word = 0
                diff_min_word = 0

            # 最长句长
            max_s=max_senctence_len(sen_list, candidate)

            # 最短句长
            min_s=min_senctence_len(sen_list, candidate)

            # 首次句子位置
            fsp =  _sen_pos[0]/len(sen_list)if len(_sen_pos)>0 else 0
            # 最后句子位置
            lsp= _sen_pos[-1]/len(sen_list)if len(_sen_pos)>0 else 0

            #出现在第一句
            oifs=occur_in_first_sentence(sen_list, candidate)
            #出现在最后一句
            oils = occur_in_last_sentence(sen_list, candidate)
            #句子出现频率
            #print("句子出现频率")
            sf = len(_sen_pos)/len(sen_list)
            #print("句子统计值")
            if len(_sen_pos)>0:
                var_sen = np.var(_sen_pos)
                mean_sen = np.mean(_sen_pos)
                skew_sen = skew(_sen_pos)
                kurt_sen = kurtosis(_sen_pos)
            else:
                var_sen = 0
                mean_sen = 0
                skew_sen = 0
                kurt_sen = 0
            #print("句差统计值")
            if len(_sen_pos)>=2:
                diff_var_sen = np.nanvar(np.diff(_sen_pos))
                diff_mean_sen = np.nanmean(np.diff(_sen_pos))
                diff_max_sen = np.nanmax(np.diff(_sen_pos))
                diff_min_sen = np.nanmin(np.diff(_sen_pos))

            else:
                diff_var_sen=0
                diff_mean_sen=0
                diff_max_sen=0
                diff_min_sen=0

            #print("skew_metrix")
            skew_metrix=skew(co_occurence_metrix[i])
            i=i+1

            _has_eng=has_eng(candidate)
            _has_digit= has_digit(candidate)

            # 相似度
            #meansim=0
            #print("相似度")
            meansim=get_mean_sim(model[candidate],vec)
            #print(model[candidate])
            #

            #介数中心性
            _degree_centrality=degree_centrality[candidate]
            _closeness_centrality = closeness_centrality[candidate]
            _betweenness_centrality=betweenness_centrality[candidate]
            _eigenvector_centrality = eigenvector_centrality[candidate]
            featrue={'头词频':hf, '词频':tf, '词长':lt,'出现在标题':oit,'首次出现词位置':fwp,'最后出现词位置':lwp,'位置跨度':maxspan,'词方差':var_word,
                           '词平均':mean_word,'词偏度':skew_word,'词峰度':kurt_word,'词差方差':diff_var_word,'词差平均':diff_mean_word,'最大词差':diff_max_word,
                           '最小词差':diff_min_word,'最长句长':max_s,'最短句长':min_s,'首次句位置':fsp,'最后句位置':lsp,'出现在第一句':oifs,'出现在最后一句':oils,'句子出现频率':sf,
                           '句方差':var_sen,'句平均':mean_sen,'句偏度':skew_sen,'句峰度':kurt_sen,'句差方差':diff_var_sen,'句差平均':diff_mean_sen,'最大句差':diff_max_sen,
                           '最小句差':diff_min_sen,'包含英文':_has_eng,'包含数字':_has_digit,'相似度':meansim,'度中心性':_degree_centrality,'接近中心性':_closeness_centrality,
                     '介数中心性':_betweenness_centrality,'特征向量中心性':_eigenvector_centrality,'共现矩阵偏度':skew_metrix,'标签':tag,'id':row[1],'关键词':candidate,'最小句中位置':w_min_s,
                     '平均句中位置':w_mean_s,'最大句中位置':w_max_s}
            featrue.update(dic)
            batch = batch.append(featrue,ignore_index=True)

            #print("----------------------------------------")

        print(lou)
        print(zong)
        for index, row in batch.iteritems():
            if index in ['词长', '词方差', '词平均', '词偏度', '词峰度', '词差方差', '词差平均', '最大词差','最小词差','最长句长','最短句长','句方差','句平均','句偏度',
                         '句峰度','句差方差','句差平均','最大句差','最小句差','共现矩阵偏度']:
                #print(batch[index])
                try:
                    batch[index] = (batch[index] - batch[index].mean()) / batch[index].std(ddof=0)
                except:
                    batch[index]=0


        res = res.append(batch, ignore_index=True)

        print('++++++++++++++++++++++++++++++')

    print(res)




    res.to_csv('tds_std.csv', header=1, index=None)




all_docs_final = pd.read_csv('new_token.csv', sep=',')

get_feature(all_docs_final)









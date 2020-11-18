'''
import numpy as np
from collections import Counter
from geatpy import crtpc
import Train.train as Train
from geatpy import bs2ri

from mutuni import mutuni
from geatpy import xovdp

from geatpy import ranking

from geatpy import selecting
from geatpy import tour
def aim(Phen):
    f = np.zeros([1,1])
    CV = np.zeros([1,1])
    for i in Phen:

        train_model=Train.train(i)
        train_model.training()

        c=Counter(i)

        f=np.append(f,np.array([[train_model.acc ]]),axis=0)
        CV=np.append(CV,np.array([[np.abs(c[1])]]),axis=0)

    return f[1:], CV[1:] # 返回目标函数值矩阵


help(crtpc) # 查看帮助
# 定义种群规模（个体数目）
Nind = 5
Encoding = 'BG' # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
# 创建“译码矩阵”


FieldD = np.array([[1]*38, # 各决策变量编码后所占二进制位数，此时染色体长度为3+2=5
                   [0]*38, # 各决策变量的范围下界
                   [1]*38, # 各决策变量的范围上界
                   [0]*38, # 各决策变量采用什么编码方式(0为二进制编码，1为格雷编码)
                   [0]*38, # 各决策变量是否采用对数刻度(0为采用算术刻度)
                   [1]*38, # 各决策变量的范围是否包含下界(对bs2int实际无效，详见help(bs2int))
                   [1]*38, # 各决策变量的范围是否包含上界(对bs2int实际无效)
                   [1]*38])# 表示两个决策变量都是连续型变量（0为连续1为离散）
# 调用crtpc函数来根据编码方式和译码矩阵来创建种群染色体矩阵
Chrom=crtpc(Encoding, Nind, FieldD)
print(Chrom)


#解码得到表现型
Phen = bs2ri(Chrom, FieldD)
print('表现型矩阵 = \n', Phen)
ObjV,CV=aim(Phen)
#目标函数
print(ObjV)
#CV矩阵
print(CV)
#计算适应度
FitnV = ranking(ObjV, CV)
print('个体的适应度为：\n', FitnV)

#选择参与进化的个体
SelCh = Chrom[selecting('tour', FitnV,Nind-1), :] # 使用'tour'锦标赛选择算子，同时片取Chrom得到所选择个体的染色体
print('选择后得到的种群染色矩阵为：\n', SelCh)

#染色体两点交叉
NewChrom = xovdp(SelCh, 1) # 设交叉概率为1
print('交叉后种群染色矩阵为：\n', NewChrom)

#染色体突变
NewChrom = mutuni('BG', SelCh, FieldD, 1)
#生成自带（精英保留策略）

bestIdx = np.argmax(FitnV)

NewChrom = np.vstack([Chrom[bestIdx, :], NewChrom])
'''
# -*- coding: utf-8 -*-
"""main.py"""
import numpy as np
from collections import Counter
import Train.train as Train
import geatpy as ea  # 导入geatpy库
import time
import math
# 创建“译码矩阵”
def sigmoid(x):
    return 1/(1+math.exp(-x))
def aim(Phen):
    f = np.zeros([1,1])
    CV = np.zeros([1,1])
    for i in Phen:

        train_model=Train.train(i)
        train_model.training()
        #print(i)
        #print(sum(i==1))
        c=sum(i == 1)
        if(c==0):
            c=1
        f=np.append(f,np.array([[train_model.acc ]]),axis=0)
        CV=np.append(CV,np.array([[math.log(c,64)]]),axis=0)

    return f[1:], CV[1:] # 返回目标函数值矩阵


FieldD = np.array([[1]*64, # 各决策变量编码后所占二进制位数，此时染色体长度为3+2=5
                   [0]*64, # 各决策变量的范围下界
                   [1]*64, # 各决策变量的范围上界
                   [0]*64, # 各决策变量采用什么编码方式(0为二进制编码，1为格雷编码)
                   [0]*64, # 各决策变量是否采用对数刻度(0为采用算术刻度)
                   [1]*64, # 各决策变量的范围是否包含下界(对bs2int实际无效，详见help(bs2int))
                   [1]*64, # 各决策变量的范围是否包含上界(对bs2int实际无效)
                   [1]*64])# 表示两个决策变量都是连续型变量（0为连续1为离散）





"""=========================遗传算法参数设置========================"""
NIND = 100;  # 种群个体数目
Encoding = 'BG'
MAXGEN = 70 # 最大遗传代数
maxormins = [-1]  # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
selectStyle = 'rws'  # 采用轮盘赌选择
recStyle = 'xovdp'  # 采用两点交叉
mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
Lind = 64 # 计算染色体长度
pc = 0.9  # 交叉概率
pm = 1 / Lind  # 变异概率
obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体
"""=========================开始遗传算法进化========================"""
start_time = time.time()  # 开始计时
Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵

help(ea.ranking)


for gen in range(MAXGEN):

    Phen = ea.bs2real(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
    ObjV, CV = aim(Phen)  # 求种群个体的目标函数值
    FitnV = ea.ranking(maxormins * ObjV,CV)  # 根据目标函数大小分配适应度值
    # 记录
    best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体

    SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]  # 选择
    SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组
    SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异
    # 把父代精英个体与子代的染色体进行合并，得到新一代种群
    Chrom = np.vstack([var_trace[gen, :], SelCh])



    print('第',gen,'代','用时：', time.time() - start_time, '秒')
# 进化完成
l=['头词频','词频','词长','TFIDF','IDF','出现在标题','首次出现词位置','最后出现词位置','位置跨度','词方差','词平均','词偏度','词峰度','词差方差','词差平均','最大词差','最小词差','最大句中位置','最小句中位置',
   '平均句中位置','最长句长','最短句长','平均句长','首次句位置','最后句位置','句跨度','出现在第一句','出现在最后一句','句子出现频率','句方差','句平均','句偏度','句峰度','句差方差','句差平均','最大句差','最小句差',
   '包含英文','相似度','度中心性','接近中心性','介数中心性','特征向量中心性','共现矩阵偏度','n','t','s','f','v','a','b','z','q','d','h','k','x','g','i','j','l','y','un','包含数字']
end_time = time.time()  # 结束计时
ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像
"""============================输出结果============================"""
best_gen = np.argmax(obj_trace[:, [1]])
print('最优解的目标函数值：', obj_trace[best_gen, 1])
variable = ea.bs2real(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
print('最优解的决策变量值为：')
n=[]
for i in range(variable.shape[1]):
    print('x' + str(i) + '=', variable[0, i])
    if(variable[0, i]):
        n.append(l[i])
print(n)
print('用时：', end_time - start_time, '秒')
import pandas as pd

data=pd.read_csv('./test.csv')
data_1=data[data['标签']==1]
data_0=data[data['标签']==0]

#print(data)
#print(data_0)
print(len(data_1))

data_0=data_0.sample(n=857,axis=0)
result=data_0.append(data_1)

result = result.sample(frac = 1)
result.to_csv('test_balance.csv',index=None)
#print(result)


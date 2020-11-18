import pandas as pd
data=pd.read_csv('./balance.csv')
data = data.fillna(0)
print(data)
data.to_csv('./balance.csv')
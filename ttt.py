import pandas as pd
data= pd.read_csv('./train.csv')
data=data.fillna(0)
data.to_csv('./train.csv',index=None)
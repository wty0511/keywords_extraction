import pandas as pd
data =pd.read_csv('./balance.csv')

for index, row in data.iteritems():
    if index in ['出现在标题', '包含英文', '包含数字', '标签', 'id', '关键词','出现在第一句','出现在最后一句']:
        continue
    print(index)
    data[index] = (data[index] - data[index].mean()) / data[index].std()
    print(data[index].mean())
data.to_csv('./balance_std.csv')
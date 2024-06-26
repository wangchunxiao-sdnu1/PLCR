# 去重 .duplicated
import pandas as pd
from sklearn import datasets
import torch
s = [1, 2, 0, 6, 3]  # [1],[2],[0],[6],[8]  01236   20143  12043
s = torch.tensor(s)
a = s.argsort().argsort()[0].item()

# A = pd.DataFrame(s, columns=['reviewerID', 'asin'])
# s1 = pd.DataFrame(s).duplicated()
# mytable = s.drop_duplicates(subset = ['cst_id'])
# print(s)
# index = s1[s1 == False].index
# print(index[0])
# indexint = []
# for item in index[0]:
#     indexint1 = int(index[0])
#     indexint.append(indexint1)
# print(indexint)
# s2=s[indexint]
# print('-----')
# # 判断是否重复
# # 通过布尔判断，得到不重复的值
#
# s_re = s.drop_duplicates()
# print(s_re)
# print('-----')
# # drop.duplicates移除重复
# # inplace参数：是否替换原值，默认False
#
# df = pd.DataFrame({'key1':['a','a',3,4,5],
#                   'key2':['a','a','b','b','c']})
# print(df.duplicated())
# print(df['key2'].duplicated())
# # Dataframe中使用duplicated
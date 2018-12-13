'''
Created by WangQL

1/23/2018
'''
import os
import pandas as pd

# read
f = open('id_series.txt', 'r')
list1 = f.readlines()
print(len(list1))
f.close()

labels_data = pd.read_csv('list.csv',index_col = 0)
print(labels_data.columns)

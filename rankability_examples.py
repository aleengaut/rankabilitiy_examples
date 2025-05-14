# -*- coding: utf-8 -*-
"""
Created on Sun Dez 15 2024

@author: Alexandre
"""
import pandas as pd
import numpy as np

# Y_{i,k} function strict count
def domYik(c):
    dj=np.zeros(len(c))
    Y=np.zeros((len(c), len(c)), dtype=float)
    i,j=0,0
    for ai in c:
        for aj in c:
            if ai>aj: 
                dj[i]=dj[i]
            if ai<aj: 
                dj[i]=dj[i]+1
                Y[i][j]=1
            i=i+1
        i=0
        j=j+1
        
    # Y: dominance count
    # dj: d element vector
    return Y, dj

# \delta_{i,k} pairwise comparison matrix
def domDeltaik(data):

    datam = np.matrix(data)
    m = datam.shape[1]
    n = datam.shape[0]
    D=np.zeros((n, n), dtype=float)
    d = np.zeros(n)
    for c in range(m):
        D = D + domYik(datam[:,c])[0]
        d = d + domYik(datam[:,c])[1]
        
    l = D//m
    l = l.transpose()
    level = [sum(li) for li in zip(*l)]
    level = np.asarray(level)
    
    # D: pairwise comparison matrix
    # d: dominance vector
    # level: dominnace level
    return D, d, level

# \delta_{i,k} pairwise comparison matrix
def domDeltaik2(data):

    datam = np.matrix(data)
    m = datam.shape[1]
    n = datam.shape[0]
    l = np.zeros(n)
    for a in range(n):
        for b in range(n):
            test = 0
            flag1=True
            flag2=True
            for c in range(m):
                if datam[a,c] > datam[b,c]: test=test+1
                if data[a,c] < data[b,c] and flag2: 
                    flag1=False
                    flag2=False
            if flag1 and test>0: test=m
            test = test // m
            l[a] = l[a] + test
    
    # D: pairwise comparison matrix
    # d: dominance vector
    # level: dominnace level
    return l

# \rho function
def rho(data):

    n = data.shape[0]
    rank = np.array(range(n))
    rank = np.flip(rank)

    # dominance vector e
    #e = domDeltaik(data)[2]
    e = domDeltaik2(data)

    rho = e.sum()/rank.sum()
    
    return round(rho, 4)

# inputs: data n X m
#d = {'col1': [5, 4, 3], 'col2': [4, 6, 3]}
#d = {'col1': [1, 2, 3, 4, 5, 6], 'col2': [6, 5, 4, 3, 2, 1]}
#d = {'col1': [5, 3, 4, 2, 1, 0], 'col2': [4, 4, 4, 2, 1, 0]} #
d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [0, 1, 2, 3, 4, 5]}
d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [0, 1, 2, 3, 4, 5], 'col3': [5, 4, 3, 2, 1, 0]}
#d = {'col1': [8, 7, 7, 5, 5, 5, 5], 'col2': [6, 6, 6, 6, 6, 6, 6]}
#d = {'col1': [5, 5, 5, 5, 5, 5], 'col2': [5, 5, 5, 5, 5, 5]} #
#d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [5, 4, 3, 2, 1, 0]} #
#d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [0, 0, 0, 0, 0, 0]} #
#d = {'col1': [5, 0, 0, 0, 0, 0], 'col2': [0, 5, 0, 0, 0, 0]} #
#d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [3, 3, 3, 2, 1, 0]} #
#d = {'col1': [1, 6, 4, 3, 2, 5], 'col2': [6, 1, 2, 5, 4, 3]} #
#d = {'col1': [5,0,1,4,3,2], 'col2': [3,3,3,2,1,0]} #
#d = {'col1': [0,5,3,2,1,4], 'col2': [5,0,1,4,3,2]} #
#d = {'col1': [5, 3, 3, 0, 0, 0]}
#d = {'col1': [0, 0, 0, 0, 0, 0]}
#d = {'col1': [5, 3, 3, 2, 1, 0]}
#d = {'col1': [1, 4, 3], 'col2': [2, 3, 2], 'col3': [3, 2, 4], 'col4': [4, 1, 1]} #
#d = {'col1': [1, 4, 3], 'col2': [2, 3, 2], 'col3': [3, 4, 4], 'col4': [4, 5, 1]} #

#d = {'col1': [5, 4, 3, 2, 1, 0], 'col2': [4, 4, 4, 2, 1, 0]} #
#d = {'col1': [2, 2, 2, 0, 0, 0, 0], 'col2': [5, 4, 4, 4, 4, 4, 4]}
#d = {'col1': [5, 5, 5, 5, 5, 5], 'col2': [0, 0, 0, 0, 0, 0]} #
#d = {'col1': [1, 1, 1, 1, 1, 1], 'col2': [4, 4, 4, 4, 4, 4]} #
#d = {'col1': [2, 0, 3, 2, 3, 1], 'col2': [4, 2, 4, 3, 3, 4]} #
#d = {'col1': [3, 2, 0, 3, 2, 4], 'col2': [2, 1, 3, 4, 3, 3]} #
#d = {'col1': [3, 1, 0, 2, 1, 0], 'col2': [5, 4, 3, 4, 4, 3]} #
#d = {'col1': [3, 2, 0, 2, 2, 3], 'col2': [3, 2, 3, 4, 3, 3]} #
#d = {'col1': [1, 1, 2, 1, 1, 2], 'col2': [3, 4, 4, 3, 4, 4]} #

df = pd.DataFrame(data=d)
data = df.copy().to_numpy()

df['d']=domDeltaik(data)[1]
#df['e']=domDeltaik(data)[2]
print(df)
print('rho:', rho(data))
print('e:', domDeltaik2(data))
print('D:', domDeltaik(data)[0])

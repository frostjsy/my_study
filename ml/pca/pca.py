# -*- coding: utf-8 -*-
"""
@time:2019-05-19

@author:jsy

"""
 
from math import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#对所有样本进行中心化（去均值操作）
def nanMean(dataMat):
    mval=np.mean(dataMat,axis=0)
    newMat=dataMat-mval
    return newMat,mval

#选择前n个特征
def select_n(vals,per):
    sort_vals=np.sort(vals)
    sort_vals=sort_vals[-1::-1]
    valSum=sum(sort_vals)
    tmp=0
    n=0
    for i in sort_vals:
        tmp+=i
        n+=1
        if(tmp>=valSum*per):
            return n

#pca降维
def pca(dataMat,per=0.9):
    newMat,mval=nanMean(dataMat)
    #计算协方差矩阵
    covMat=np.cov(newMat,rowvar=0)
    #计算协方差矩阵的特征值和特征向量
    vals,vecs=np.linalg.eig(np.mat(covMat))
    #选取特征向量维数
    n=select_n(vals,per)
    print("需要的维数：",n)
    #特征值进行排序
    vals_index=np.argsort(vals)
    #最大n个值的下标
    n_vals_index=vals_index[-1:-(n+1):-1]
    #最大n个特征值对应的特征向量
    n_vecs=vecs[:,n_vals_index]
    #保留低维空间的数据
    lowMat=newMat*n_vecs
    #重构特征向量
    reMat=lowMat*n_vecs.T+mval
    return lowMat,reMat

if __name__ == '__main__':
    data = np.random.randint(1,10,size = (3,5))
    print( data)
    lowMat,reMat= pca(data,0.9)
    print(lowMat)
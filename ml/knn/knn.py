# -*- coding: utf-8 -*-
"""
@time:2019-04-21

@author:jsy

"""


import numpy as np
import operator
from os import listdir

#KNN分类
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #计算向量每一维度的差异
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    #计算要预测的样本与标签样本之间的距离
    distances = sqDistances**0.5
    
    #获取各距离下标
    sortedDistIndicies = distances.argsort()  
    #print(sortedDistIndicies)
    classCount={}  
    #获取k近邻        
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]

#加载数据
def createDataSet():
    group = np.array([[1.0,1.2],[1.1,1.2],[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','A','A','B','B']
    return group, labels

if __name__=='__main__':
    group,labels=createDataSet()
    #print(group,labels)
    print(classify0([0,0],group,labels,3))


# -*- coding: utf-8 -*-
"""
@time:2019-05-04

@author:jsy

"""

from numpy import *

'''获取数据'''
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    
    return postingList,classVec
                 
'''从语料库，获取词列表'''
def get_vocs(data):
    vocs=set()
    for doc in data:
        vocs=vocs.union(set(doc))
    return list(vocs)

'''获取one-hot特征'''
def get_onehot(words,vocs):
    res=[0]*len(vocs)
    for word in words:
        if word in vocs:
            res[vocs.index(word)]=1
    return res

'''获取先验概率'''
def get_prior_prob(train,label):
    num_train=len(train)
    num_vec=len(train[0])
    p1=sum(label)/float(num_train)
    p0_num=ones(num_vec)
    p1_num=ones(num_vec)
    p1_denom=2.0
    p0_denom=2.0
    for i in range(num_train):
        if label[i]==1:
            p1_num+=train[i]
            p1_denom+=sum(train[i])
        else:
            p0_num+=train[i]
            p0_denom+=sum(train[i])
    p1_vec_prob=log(p1_num/p1_denom)
    p0_vec_prob=log(p0_num/p0_denom)
    return p0_vec_prob,p0_vec_prob,p1

'''分类'''
def classify(vec,p0_vec,p1_vec,p1_prior):
    p1=sum(vec*p1_vec)+log(p1_prior)
    p0=sum(vec*p0_vec)+log(1-p1_prior)
    if p1 > p0:
        return 1
    else: 
        return 0
    
if __name__=="__main__":
    train,labels = loadDataSet()
    vocs =get_vocs(train)
    
    print(vocs)
    trainMat=[]
    for doc in train:
        trainMat.append(get_onehot(doc,vocs))
    
    
    p0_vec,p1_vec,p1_prior=get_prior_prob(array(trainMat),array(labels))
    
    test=['love', 'my', 'dalmation']
    print(classify(get_onehot(test,vocs),p0_vec,p1_vec,p1_prior))
    
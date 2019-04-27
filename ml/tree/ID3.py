# -*- coding: utf-8 -*-
"""
@time:2019-04-27

@author:jsy

"""

from math import log
import operator

#计算香农熵
def calcEnt(data):
    numEnt=len(data)
    labelCounts={}
    for featVec in data:
        label=featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label]=0
        labelCounts[label]+=1
        ent=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEnt
        ent-=prob*log(prob,2)
    return ent

#划分数据集
def splitData(data, axis, value):
    retData = []
    for featVec in data:
        #去掉已选属性所对应列的值
        if featVec[axis] == value:
            reducedVec = featVec[:axis]     
            reducedVec.extend(featVec[axis+1:])
            retData.append(reducedVec)
    return retData

#选择最好的划分特征
def chooseBestFeature(data):
    numFea = len(data[0]) - 1      
    baseEnt = calcEnt(data)
    bestGain = 0.0; 
    bestFea = -1
    for i in range(numFea):
        #获取属性列
        featValues = [example[i] for example in data]
        #获取属性i的所有对应值
        uniqueVals = set(featValues)       
        newEnt = 0.0
        #计算属性i对应的信息熵
        for value in uniqueVals:
            subData= splitData(data, i, value)
            prob = len(subData)/float(len(data))
            newEnt += prob * calcEnt(subData)     
            curGain = baseEnt - newEnt
        #选择最好的信息熵
        if (curGain > bestGain):      
            bestGain = curGain         
            bestFea = i
    return bestFea                    

#获取最大类别数
def maxCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#构建决策树
def createTree(data,feaLabels):
    classList = [example[-1] for example in data]
    
    #当所有的类别相等时，停止划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    
    #当没有数据中没有属性时，停止划分
    if len(data[0]) == 1: 
        return maxCnt(classList)
    
    #获取初始的第一个划分属性
    bestFeat = chooseBestFeature(data)
    bestFeatLabel = feaLabels[bestFeat]
    
    #决策树结构：字典
    myTree = {bestFeatLabel:{}}
    
    #去掉已被选择了的属性
    del(feaLabels[bestFeat])
    featValues = [example[bestFeat] for example in data]
    uniqueVals = set(featValues)
    
    #根据属性值，继续用剩下的属性对数据集进行划分
    for value in uniqueVals:
        #subLabels = feaLabels[:]    
        #myTree[bestFeatLabel][value] = createTree(splitData(data, bestFeat, value),subLabels)
        myTree[bestFeatLabel][value] = createTree(splitData(data, bestFeat, value),feaLabels)
    return myTree                            
    
def classify(inputTree,feaLabels,testVec):
    firstFea = list(inputTree.keys())[0]
    secondDict = inputTree[firstFea]
    #print(feaLabels)
    featIndex = feaLabels.index(firstFea)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat,feaLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

if __name__=="__main__":
    data= [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feaLabels=['A','B']
    decisionTree=createTree(data,feaLabels)
    print(classify(decisionTree,['A','B'],[0,1]))

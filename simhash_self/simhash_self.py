# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:05:47 2019

@author: jsy
"""
import jieba
import jieba.analyse
import hashlib
import pandas as pd

class Simhash_self:
    def __init__(self,f=64,text=None):
        self.f=f
        self.value=self.get_simhash(text)
    
    #获取simhash值
    def get_simhash(self,text):
        res=None
        word_weight=self.tokenize(text)
        res=self.cal_simhashValue(word_weight)
        return res
    
    #分词，加权
    def tokenize(self,text):
        seg=jieba.cut(text)
        jieba.analyse.set_stop_words('stop_words_1893.txt')
        word_weight=jieba.analyse.extract_tags('|'.join(seg),topK=20, withWeight=True)
        return word_weight
        
    #hash加密
    def hashfunc(self,x):
        return int(hashlib.md5(x).hexdigest(), 16)

    #计算simhash值
    def cal_simhashValue(self,word_weight):
        v=[0]*self.f
        masks = [1 << i for i in range(self.f)]
       
        for word,weight in word_weight:
            h=self.hashfunc(word.encode('utf-8'))
            w=weight
            for i in range(self.f):
                v[i]+=w if h&masks[i] else -w
        ans=0
        
        for i in range(self.f):
            if v[i]>0:
                ans|=masks[i]
        return ans
    
    #计算海明距离
    def get_distance(self,another):
        assert self.f==another.f
        x=(self.value^another.value)&((1 << self.f)-1)
        res=0
        
        #计算一个数中1的个数
        while x:
            res+=1
            x&=x-1
        return res

if __name__=='__main__':
    texts=pd.read_excel('1.xlsx',columns=['comment'],encoding='utf-8')
    #print(texts.columns)
    #print(texts['comment'])
    texts['simhash_value']=texts.apply(lambda row:Simhash_self(64,row['comment']), axis=1)
    
    print(texts.loc[1,'simhash_value'].value)
    print(texts.loc[1,'simhash_value'].get_distance(texts.loc[1,'simhash_value']))
    print(texts.loc[0,'simhash_value'].get_distance(texts.loc[1,'simhash_value']))
    print(texts.loc[6,'simhash_value'].get_distance(texts.loc[7,'simhash_value']))
    print(texts.loc[60,'simhash_value'].get_distance(texts.loc[61,'simhash_value']))



    

# -*- coding: utf-8 -*-
"""
@time:2019-05-12

@author:jsy
"""
import math
import jieba
import numpy as np

def tf(word,doc):
    tf_dict={}
    for w in doc:
        cnt=tf_dict.get(w,0)+1
        tf_dict[w]=cnt
    return tf_dict[w]
    
def docs_contains_word(word, D):
  return sum(1 for d in D if word in d)
 
def idf(word, D):
  return math.log(len(D)/(1 + docs_contains_word(word, D)))
 
def tfidf(word,doc,D):
  return tf(word, doc) * idf(word, D)

if __name__=="__main__":

    #语料库
    Docs =["非关癖爱轻模样，冷处偏佳。别有根芽，不是人间富贵花。",
           "谢娘别后谁能惜，漂泊天涯。寒月悲笳，万里西风瀚海沙。",
           "彤云久绝飞琼字，人在谁边。人在谁边，今夜玉清眠不眠。",
           "香销被冷残灯灭，静数秋天。静数秋天，又误心期到下弦。"]
    #词库
    D=[]
    for doc in Docs:
        d=jieba.cut(doc)
        D.append(list(d))
          
    print(D)
    print(tfidf(D[0][0],D[0],D))




    

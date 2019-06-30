#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 13:37:05 2019

@author: jsy
"""
'''
找到一个句子里面重复的子串
'''
import time
import re

def slice_str(s,w):
    '''滑窗切片返回结果'''
    seq=[]
    for i in range(len(s)-w+1):
        seq.append(s[i:i+w])
    return seq
    
def get_repeat_seq(s):
    '''统计重复串数量'''
    subseq_num={}
    subseqs=[]
    for i in range(3,min(6,int(len(s)/2))):
        '''获取i长度的子串'''
        subseqs+=slice_str(s,i)
    
    #print(subseqs)
    '''统计子串长度'''
    for i in range(len(subseqs)):
        print(subseqs[i])
        print(subseq_num.get(subseqs[i],0))
        subseq_num[subseqs[i]]=subseq_num.get(subseqs[i],0)+1
    #print(subseq_num)
    sort_subseqs=sorted(subseq_num.items(),key=lambda e:e[1],reverse=True)
    return sort_subseqs[0]

'''测试'''
p0=time.time()
s='我是中国人，中国人，我是中国人，祖国，我爱我的祖国；中文好难，中文好难'
s=re.sub(r'[^\w]','',s)
print(get_repeat_seq(s))
p1=time.time()
print(p1-p0)
        
        
        
        
        
        
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:18:08 2019

@author: jsy

"""

'''
正则表达式应用汇总
match findall search sub split常用函数的应用
'''
import re

'''
match使用
mtach匹配字符串开头的零个或多个字符。
'''

s1='123go! 极限挑战，挑战自己,666!'
s2='hi, baby,123go! 极限挑战，挑战自己,666!'
p=re.compile('\d+')
c1=p.match(s1)
print(c1,'\n',c1.group()) #s1的开头匹配到了p,c1.group()结果为123

c2=p.match(s2)
print(c2) #s2开头未匹配p,c2的结果为空


"""
findall使用
返回字符串中所有非重叠匹配项的列表。
"""
c3=p.findall(s1)
print(c3)   #结果为['123', '666']
c4=p.findall(s2)
print(c4) #结果为['123', '666'],不受开头限制

"""
search使用
扫描字符串，寻找与模式匹配的字符串，然后返回
匹配对象，如果没有找到匹配，则为None。
"""
c5=p.search(s1)
print(c5.group()) #输出123

"""
sub使用
扫描字符串，寻找与模式匹配的字符串，然后返回
匹配对象，如果没有找到匹配，则为None。
"""
c6=re.sub(p,'',s1) #将数字替换为''
print(c6) #输出结果为：go! 极限挑战，挑战自己,!

"""
split使用
根据模式的出现情况拆分源字符串，返回包含结果子字符串的列表。如果
捕获括号在pattern中使用，然后是all的文本
模式中的组也作为结果的一部分返回
列表。如果maxsplit不为零，则最多发生maxsplit拆分，
字符串的其余部分作为最后一个元素返回
的列表。
"""
c6=re.split(re.compile('!|，|\d+'),s1) #以'!''，''\d+'分割s1字符串
print(c6) #输出结果为：['', 'go', ' 极限挑战', '挑战自己,', '', '']
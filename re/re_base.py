#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:59:50 2019

@author: jsy
"""

import re
"""
一些常见的正则表达式实现
"""
'''
\将下一个字符标记为一个特殊字符、或一个原义字符、或一个向后引用、或一个八进制转义符。例如，“n"匹配字符"n"。"\n"匹配一个换行符。串行"\\"匹配"\"而"\("则匹配"("。
'''
p1=re.compile('n')
p2=re.compile('\(')
s='asdfngkl\('
c1=p1.findall(s)
print(c1) #输出['n']
c2=p2.findall(s)
print(c2) #输出['(']

'''
^匹配输入字符串的开始位置。
$匹配输入字符串的结束位置。
'''
p3=re.compile('^as')
p4=re.compile('as$')
s1='asdfgh;;;vbn'
s2='dfgh;mnas'
c3=p3.findall(s1)
print(c3) #输出：['as']
c3=p3.findall(s2)
print(c3) #输出:[]

c4=p4.findall(s1)
print(c4) #输出:[]
c4=p4.findall(s2)
print(c4) #输出：['as']

'''
贪婪匹配模式
*匹配前面的子表达式零次或多次，等价于{0,}
+匹配前面的子表达式一次或多次，等价于{1,}
?匹配前面的子表达式零次或一次，等价于{0,1}
'''
s1='1'
s2='10000'
p1=re.compile('10*') #0出现0次或多次
p2=re.compile('10+') #0出现1次或多次
p3=re.compile('10?') #0出现0次或1次
print(p1.findall(s1)) #输出：['1']
print(p1.findall(s2)) #输出：['10000']
print(p2.findall(s1)) #输出：[]
print(p2.findall(s2)) #输出：['10000']
print(p3.findall(s1)) #输出： ['1']
print(p3.findall(s2)) #输出：['10']

'''
贪婪匹配模式
{n}:n为非负整数，匹配固定出现的n次数
{n,}:n为非负整数，匹配至少出现n次
{n,m}:n,m为非负整数,n<m，匹配出现n-m次
'''
s1='1  10 10000 1000000 1000000000'
p1=re.compile('10{4}')  #0出现4次
p2=re.compile('10{1,}') #0出现至少1次 
p3=re.compile('10{2,6}')   #0出现2-6次

print(p1.findall(s1)) #输出：['10000', '10000', '10000']
print(p2.findall(s1)) #输出：['10', '10000', '1000000', '1000000000']
print(p3.findall(s1)) #输出: ['10000', '1000000', '1000000']

'''
当?出现在*,+,?，{n}，{n,}，{n,m}）后面时，匹配模式是非贪婪的。非贪婪模式尽可能少的匹配所搜索的字符串，，而默认的贪婪模式则尽可能多的匹配所搜索的字符串。
'''
s1='1  10 10000 1000000 1000000000'
p1=re.compile('10*?')  #0出现4次
p2=re.compile('10{1,}?') #0出现至少1次 
p3=re.compile('10{2,6}?')   #0出现2-6次

print(p1.findall(s1)) #输出：['1', '1', '1', '1', '1']
print(p2.findall(s1)) #输出：['10', '10', '10', '10']
print(p3.findall(s1)) #输出: ['100', '100', '100']



'''
.匹配除\n以外的任意字符
'''
p=re.compile('as.*df')
s1='asjkljldf'
s2='asjkdfkllkll'
print(p.findall(s1)) #输出：['asjkljldf']
print(p.findall(s2)) #输出：['asjkdf']


'''
(pattern)匹配到()里面的pattern
(?:pattern)取消组，匹配pattern
'''
p=re.compile('(z|f)ood') #只能匹配得到z或者f
p.findall('food') #输出：['f']

p=re.compile('z|food') #匹配得到z或者food
p.findall('food') #输出：['food']

p=re.compile('(?:z|f)ood') #匹配得到zood或者food
p.findall('food') #输出：['food']


'''
断言：
(?=pattern)正向肯定预查，在任何匹配pattern的字符串开始处匹配查找字符串
(?!pattern)正向否定预查，在任何不匹配pattern的字符串开始处匹配查找字符串
(?<=pattern)反向肯定预查，与正向肯定预查类拟，只是方向相反；且pattern需为固定长度
(?<!pattern)反向否定预查，与正向否定预查类拟，只是方向相反；且pattern需为固定长度
'''
s1='山清水秀'
s2='山高水远'
p1=re.compile('水(?=清|秀|冰冷)')
print(p1.findall(s1)) #输出: ['水']
print(p1.findall(s2)) #输出: []

p1=re.compile('水(?!清|秀|冰冷)')
print(p1.findall(s1)) #输出: []
print(p1.findall(s2)) #输出: ['水']

#p1=re.compile('(?<=清|秀|冰冷)水') #会报错 error:look-behind requires fixed-width pattern
p1=re.compile('(?<=清|秀|冷)水')
print(p1.findall(s1)) #输出: ['水']
print(p1.findall(s2)) #输出: []

p1=re.compile('(?<!清|秀|冷)水')
print(p1.findall(s1)) #输出: []
print(p1.findall(s2)) #输出: ['水']

'''
[xyz]:字符集合。匹配所包含的任意一个字符。
[^xyz]:负值字符集合。匹配未包含的任意字符。
[a-z]:字符范围。匹配指定范围内的任意字符。
[^a-z]:负值字符范围。匹配任何不在指定范围内的任意字符。
'''
s='my dog. He is cute.'
p1=re.compile('[hou]',re.I)#匹配hou任意一个字符，且忽略大小写
print(p1.findall(s)) #输出：['h', 'o', 'H', 'u']

p1=re.compile('[^hou]',re.I)#匹配非hou任意一个字符，且忽略大小写
print(p1.findall(s)) #输出：['T', 'i', 's', ' ', 'i', 's', ' ', 'm', 'y', ' ', 'd', 'g', '.', ' ', 'e', ' ', 'i', 's', ' ', 'c', 't', 'e', '.']

p1=re.compile('[a-g]',re.I)#匹配a-g任意一个字符，且忽略大小写
print(p1.findall(s)) #输出：['d', 'g', 'e', 'c', 'e']

p1=re.compile('[^a-g]',re.I)#匹配任意一个字符，且忽略大小写
print(p1.findall(s)) #输出：['m', 'y', ' ', 'o', '.', ' ', 'H', ' ', 'i', 's', ' ', 'u', 't', '.']

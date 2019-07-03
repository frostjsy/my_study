#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:23:48 2019

@author: jsy
"""

def chinese2digits(uchars_chinese):
    common_used_numerals = {
        '零': 0,
        '一': 1,
        '二': 2,
        '两': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9,
        '十': 10,
        '百': 100,
        '千': 1000,
        '万': 10000,
        '亿': 100000000,
        'k': 1000,
        'W': 10000,
        'K': 1000,
        'w': 10000
    }
    total = 0
    r = 1  # 表示单位：个十百千...
    
    #从头开始遍历中文字符串
    for i in range(len(uchars_chinese) - 1, -1, -1):
        #获取字符串对应的值
        val = common_used_numerals.get(uchars_chinese[i])
        #查看val是不是位权‘十、百、千、万、亿’
        if val >= 10 and i == 0:            #存在十五，十八这样的数
            if val > r:
                r = val
                total = total + val
            else:                          #更新位权表示r
                r = r * val
                
        elif val >= 10:                   #中文数字为‘十、百、千、万、亿’
            if val > r:                   #例如小于1w时，对于r的操作
                r = val
            else:                         #大于1w时，对于r的操作 
                r = r * val
        else:                             #为数字时，进行计算
            total = total + r * val
    return total
test_dig = ['八','五十',
                '十一',
                '一百二十三',
                '一千二百零三',
                '一万一千一百零一',
                '十万零三千六百零九',
                '一百二十三万四千五百六十七',
                '一千一百二十三万四千五百六十七',
                '一亿一千一百二十三万四千五百六十七',
                '一百零二亿五千零一万零一千零三十八']

for t in test_dig:
    print(chinese2digits(t))
    

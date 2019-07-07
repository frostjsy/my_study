#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:25:39 2019

@author: jsy
"""


from tensorflow import keras
from tensorflow.keras import layers
import re
import os
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model,save_model
from sklearn.metrics import accuracy_score


def get_text():
    '''获取文本数据'''
    data_dir = 'temp'
    data_file = 'text_data.txt'
    with open(os.path.join(data_dir, data_file), 'r') as f:
        text = f.read()
    return text

    
def clean_text(text_string):
    ''''清洗数据'''
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = ' '.join(text_string.split())
    text_string = text_string.lower()
    #print(text_string)
    return (text_string)

def get_vocab(text,freq=3):
    '''获取词表'''
    words = text.split()
    # 删除低频词，减少噪音影响,出现频次大于10
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    vocab = set(trimmed_words)
    vocab_to_int = {w: (c+1) for c, w in enumerate(vocab)}
    int_to_vocab = {(c+1): w for c, w in enumerate(vocab)}  
    return vocab,vocab_to_int,int_to_vocab

def preocess_data(vocab_to_int,words):
    '''对原文本进行vocab到int的转换'''
    x = [vocab_to_int[w] for w in words if w in vocab_to_int]
    
    return x

def pad_text(x,maxlen):
    '''使用keras提供的pad_sequences来将文本pad为固定长度'''
    x= keras.preprocessing.sequence.pad_sequences(x, maxlen, padding='post')
    return x


def lstm_model(input_dim,maxlen):
    '''构建lstm模型'''
    model = keras.Sequential([
        layers.Embedding(input_dim=input_dim, output_dim=32, input_length=maxlen),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def train(x_train, y_train,model):
    '''训练'''
    model.summary()
    history = model.fit(x_train, y_train, batch_size=64, epochs=2,validation_split=0.2)
    #model.save('my_model.h5')
    save_model(model,'my_model.h5')

def predict(x_test,y_test):
    '''预测'''
    new_model = load_model('my_model.h5')
    new_prediction = new_model.predict(x_test)
    class_pred=new_model.predict_classes(x_test)
    print(accuracy_score(y_test,class_pred))
    
    

if __name__=='__main__':  
    text=get_text()
    text_data=text.split('\n')
    text_data=[x.split('\t') for x in text_data if len(x) >= 1]
    text_data=np.array(text_data)
    '''获取数据和标签'''
    text_label=text_data[:,0]
    text_data=text_data[:,1]
    
    text=clean_text(text)
    vocab,vocab_to_int,int_to_vocab=get_vocab(text)
    
    input_dim=len(vocab)+1
    maxlen=64
    
    text_x=[]
    text_y=np.array([1 if x=='ham' else 0 for x in text_label])
    for line in text_data:
        line=preocess_data(vocab_to_int,clean_text(line).split())
        text_x.append(line)
    text_x=np.array(text_x)
    print(text_x.shape)
    text_x=pad_text(text_x,maxlen)
    print(text_x.shape)
    x_train,x_test,y_train,y_test=train_test_split(text_x,text_y,test_size=0.2,random_state=0)
    model = lstm_model(input_dim,maxlen)

    #train(x_train, y_train,model)
    predict(x_test,y_test)
    
    
    
    
    
        
    
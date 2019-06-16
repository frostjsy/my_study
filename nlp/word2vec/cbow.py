#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:18:37 2019

@author: jsy
"""

import numpy as np
import tensorflow as tf
import random
from collections import Counter
import collections
import math


with open('data/text8') as f:
    text = f.read()
    
    
# 定义函数来完成数据的预处理，包括：替换文本中特殊符号并去除低频词，对文本分词，构建语料，单词映射表
def preprocess(text, freq=5):
    '''
    对文本进行预处理
    
    参数
    ---
    text: 文本数据
    freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    return trimmed_words


# 清洗文本并分词
words = preprocess(text)
print(words[:20])

vocabulary_size = 50000

#创建数据集
def build_dataset(words):
    # UNK表示不再字典中的词
    count = [['UNK', -1]]
    # 返回vocabulary_size个(word,count)的元组
    count.extend(Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    #给单词标号
    for word, _ in count:
        dictionary[word] = len(dictionary)
    #data存储词的编号来替换原始词
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  

data_index = 0
#生成batch
def generate_batch(batch_size, skip_window):
    # skip window为窗口大小
    global data_index

    span = 2 * skip_window + 1 #left word right

    batch = np.ndarray(shape=(batch_size,span-1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    #添加和删除在最后节点的队列
    buffer = collections.deque(maxlen=span)

    #循环获取窗口大小个词
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size):
        target = skip_window  #窗口中心词
        col_idx = 0
        for j in range(span):
            if j==span//2:
                continue
            batch[i,col_idx] = buffer[j] 
            col_idx += 1
        labels[i, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    assert batch.shape[0]==batch_size and batch.shape[1]== span-1
    return batch, labels

print('data:', [reverse_dictionary[di] for di in data[:8]])


num_steps = 100001

if __name__ == '__main__':
    batch_size = 128
    embedding_size = 128 #词嵌入的维度
    skip_window = 1 # 窗口大小，左右窗口大小

    valid_size = 16 # 随机选16个词
    valid_window = 100 
    #随机选16个单词
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,random.sample(range(1000,1000+valid_window), valid_size//2))
    num_sampled = 64 #负采样数

    graph = tf.Graph()

    with graph.as_default():

        #输入占位符
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*skip_window])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # 词嵌入声明
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        #权重w
        softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                         stddev=1.0 / math.sqrt(embedding_size)))
        #偏量b
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        #输入窗口的词向量求平均值，得到新的输入
        embeds = None
        for i in range(2*skip_window):
            embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
            print('embedding %d shape: %s'%(i,embedding_i.get_shape().as_list()))
            emb_x,emb_y = embedding_i.get_shape().as_list()
            if embeds is None:
                embeds = tf.reshape(embedding_i,[emb_x,emb_y,1])
            else:
                embeds = tf.concat([embeds,tf.reshape(embedding_i,[emb_x,emb_y,1])],2)

        assert embeds.get_shape().as_list()[2]==2*skip_window
        print("Concat embedding size: %s"%embeds.get_shape().as_list())
        avg_embed =  tf.reduce_mean(embeds,2,keep_dims=False)
        print("Avg embedding size: %s"%avg_embed.get_shape().as_list())
        loss = tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, train_labels, avg_embed,num_sampled, vocabulary_size)    
        cost = tf.reduce_mean(loss)
        #定义优化器
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(cost)

        # 计算minbatch与所有嵌入词之间的cosine相似性
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            batch_data, batch_labels = generate_batch(batch_size, skip_window)
            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, cost], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    #2000 batches计算一次平均损失
                print("steps {}:loss{}".format(step, average_loss))
                average_loss = 0
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8 #取出前8个最相似的词
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = normalized_embeddings.eval()

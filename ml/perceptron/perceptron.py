#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 19:50:22 2019

@author: jsy
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#获取所有的数据集
mnist_data = input_data.read_data_sets("/MNIST_data",one_hot=True)
#创建一个交互式的会话
sess = tf.InteractiveSession()

'''
构造一个两层的神经网络，输入是一个28×28×1=784维的向量，第一层是有784×300个节点
和300个偏置，然后通过relu激活函数使用dropout将部分节点失效，最后输出一个300维的
向量。第二层有300×10个节点和10个偏置，第二层使用relu激活函数和dropout，输出一个
10维向量。最后，再通过softmax层，输出属于某个数字的概率。

'''
#定义神经网络的参数
in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units],dtype=tf.float32))
w2 = tf.Variable(tf.truncated_normal([h1_units,10],stddev=0.1),dtype=tf.float32)
b2 = tf.Variable(tf.zeros([10],dtype=tf.float32))
#定义输入变量
x = tf.placeholder(dtype=tf.float32,shape=[None,in_units])
#定义dropout保留的节点数量
keep_prob = tf.placeholder(dtype=tf.float32)
#定义前向传播过程
h1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
#使用dropout
h1_drop = tf.nn.dropout(h1,keep_prob)
#定义输出y
y = tf.nn.softmax(tf.matmul(h1_drop,w2)+b2)
#定义输出变量
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])


#定义损失函数
loss_func = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(loss_func)

#初始化变量
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys = mnist_data.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
#计算准确率
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,dtype=tf.float32))
#训练集上的准确率
print("train accuracy:",accuracy.eval({x:mnist_data.train.images,y_:mnist_data.train.labels,keep_prob:1.0}))
print("test accuracy:",accuracy.eval({x:mnist_data.test.images,y_:mnist_data.test.labels,keep_prob:1.0}))


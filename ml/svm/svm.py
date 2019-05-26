#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:42:46 2019

@author: jsy
"""
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class my_svm:
    def __init__(self,kernel = 'lin1' ):
        self.kernel = kernel
        self.ploy_d=5
        self.rbf_sigma = -1.0
        self.tanh_beta = 0.9
        self.tanh_theta = -0.6
        self.maxtimes=10
        self.lin1_C=0.09


    def train_svm(self, train_x, train_label, test_x):

        # 创建会话
        sess = tf.Session()

        # 训练数据占位符
        x_data = tf.placeholder(shape=[None, train_x.shape[1]], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, train_label.shape[1]], dtype=tf.float32)

        # 预测数据占位符
        prexdata = tf.placeholder(shape=[None, train_x.shape[1]], dtype=tf.float32)

        # 线性可分情况
        if self.kernel == 'lin1':
            # 线性可分：变量
            W = tf.Variable(tf.random_normal(shape=[train_x.shape[1], train_label.shape[1]]), dtype=tf.float32)
            b = tf.Variable(tf.random_normal(shape=[1, train_label.shape[1]]), dtype=tf.float32)

            # 分割线的值
            model_output = tf.subtract(tf.matmul(x_data, W), b)

            # L2范数
            l2_term = tf.reduce_sum(tf.square(W))

            # 成本函数
            class_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
            loss = tf.add(class_term, tf.multiply(self.lin1_C, l2_term))

            # 计算预测的值
            prediction = tf.squeeze(tf.sign(tf.subtract(tf.matmul(prexdata, W), b)))

        # 核函数情况
        else:
            # 其实就是拉格朗日因子变量
            Lagrange = tf.Variable(tf.random_normal(shape=[1, train_x.shape[0]]))  # 和样本的个数是一致的

            # linear 线性核函数
            if self.kernel == 'linear':
                # 计算核函数值
                kernel_num = tf.matmul(x_data, tf.transpose(x_data))
                # 预测函数
                pred_num = tf.matmul(x_data, tf.transpose(prexdata))

            elif self.kernel == 'poly':
                # 计算核函数值
                kernel_num = tf.pow(tf.matmul(x_data, tf.transpose(x_data)), self.ploy_d)
                # 预测函数
                pred_num = tf.pow(tf.matmul(x_data, tf.transpose(prexdata)), self.ploy_d)

            elif self.kernel == 'sigmoid':
                # 计算核函数值
                kernel_num = tf.tanh(self.tanh_beta * tf.matmul(x_data, tf.transpose(x_data)) + self.tanh_theta)
                # 预测函数
                pred_num = tf.tanh(self.tanh_beta * tf.matmul(x_data, tf.transpose(prexdata)) + self.tanh_theta)

            elif self.kernel == 'rbf':
                # 计算核函数的值，将模的平方展开：a方+b方-2ab
                xdatafang = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
                momo = tf.add(tf.subtract(xdatafang, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
                                  tf.transpose(xdatafang))
                kernel_num = tf.exp(tf.multiply((1 / (-2 * tf.pow(self.rbf_sigma, 2))), tf.abs(momo)))

                # 计算预测函数的值，将模的平方展开：a方+b方-2ab
                xfang = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
                prefang = tf.reshape(tf.reduce_sum(tf.square(prexdata), 1), [-1, 1])
                mofang = tf.add(tf.subtract(xfang, tf.multiply(2., tf.matmul(x_data, tf.transpose(prexdata)))),
                                    tf.transpose(prefang))
                pred_num = tf.exp(tf.multiply((1 / (-2 * tf.pow(self.rbf_sigma, 2))), tf.abs(mofang)))
            else:
                print('核函数命名错误')
                kernel_num = 0
                pred_num = 0
                import time
                time.sleep(1000)

            # 计算成本函数
            # 第一项拉格朗日因子的和
            sum_alpha = tf.reduce_sum(Lagrange)
            # 计算第二项
            la_la = tf.matmul(tf.transpose(Lagrange), Lagrange)
            y_y = tf.matmul(y_target, tf.transpose(y_target))
            second_term = tf.reduce_sum(tf.multiply(kernel_num, tf.multiply(la_la, y_y)))
            # 最终的
            loss = tf.negative(tf.subtract(sum_alpha, second_term))

            # 计算预测的类别以及正确率
            prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), Lagrange), pred_num)
            # 类别输出，shape=（样本数,）
            prediction = tf.squeeze(tf.sign(prediction_output))

        # 调用优化器
        my_opt = tf.train.GradientDescentOptimizer(0.004)  # 学习率
        train_step = my_opt.minimize(loss)

        # 初始化变量
        init = tf.global_variables_initializer()
        sess.run(init)

        # 开始训练
        loss_vec = []  # 存储每一次的误差

        #  属性数据shape = (样本数，单个样本特征数)
        #  标签数据shape = (样本数，1)
        labely = train_label # 更改维度
        print(type(train_x))
        print(type(labely))
        for i in range(self.maxtimes):
            # 训练
            sess.run(train_step, feed_dict={x_data: train_x, y_target: labely})  # 全部样本一齐训练
            # 获得误差
            temp_loss = sess.run(loss, feed_dict={x_data: train_x, y_target: labely})
            loss_vec.append(temp_loss)

        # 获得拉格朗日因子的值
        # 因为当kenerl的值为lin1的时候，没有定义拉格朗日
        try:
            lan = Lagrange.eval(session=sess)
        except UnboundLocalError:
            pass

        # 输出预测的类别
        # 训练数据
        trlas = sess.run(prediction, feed_dict={x_data: train_x, y_target: labely, prexdata: train_x})
        # 预测数据
        prelas = sess.run(prediction, feed_dict={x_data: train_x, y_target: labely, prexdata: test_x})
        # print(trlas)
        #print(Counter(trlas))
        print(prelas)
        # 返回训练误差，训练输出，预测输出
        return loss_vec, trlas, prelas


if __name__=="__main__":
    #设置产生100条样本
    n=100
    #设置x为数据输入，其形状为100×3 的array
    x_data=np.random.uniform(low=-5,high=5,size=(n,2))
    print(x_data)
    #计算公式化的数据y输出
    y_data=np.dot(x_data,np.array([[1],[-1]]))

    #y_data[y_data>=0 and y_data<=5]=1
    y_data[y_data>0]=1
    y_data[y_data<0]=-1
    #y_data[y_data%1!=0]=1
    #这里将y_data的array数据修改为one-hot编码，编码后通过toarray的方法返回一个one-hot类型的数组
    #y_data=OneHotEncoder(sparse=True).fit_transform(y_data).toarray()
    #print(y_data)
    svm=my_svm()
    #standar_scaler = StandardScaler()
    standar_scaler = MinMaxScaler()
    
    x_data=standar_scaler.fit_transform(x_data)
    print(x_data)

    x_train=x_data[0:int(0.8*len(x_data))]
    x_test=x_data[int(0.8*len(x_data)):len(x_data)]
    y_train=y_data[0:int(0.8*len(x_data))]
    svm.train_svm(x_train,y_train,x_test)

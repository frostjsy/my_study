# -*- coding: utf-8 -*-
"""
@time:2019-06-01

@author:jsy

"""

from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import tensorflow as tf


#设置产生500条样本
n=100
#设置x为数据输入，其形状为500×3 的array
x_data=np.random.uniform(low=-5,high=5,size=(n,3))
print(x_data)
#计算公式化的数据y输出
y_data=np.dot(x_data,np.array([[7],[-4],[-1]]))

#y_data[y_data>=0 and y_data<=5]=1
y_data[y_data>5]=2
y_data[y_data<0]=0
y_data[y_data%1!=0]=1
#这里将y_data的array数据修改为one-hot编码，编码后通过toarray的方法返回一个one-hot类型的数组
y_data=OneHotEncoder(sparse=True).fit_transform(y_data).toarray()


#x,y定义为占位符，作为模型的输入接口
#根据所构建的数据集来定义占位符的shape为（None,3）
x=tf.placeholder(tf.float32,[None,3],'x_ph')
y=tf.placeholder(tf.float32,[None,3],'y_ph')

#权重和偏执项的节点构建
w=tf.Variable(tf.ones([3,3]),dtype=tf.float32,name='w')
b=tf.Variable(tf.ones([3]),dtype=tf.float32,name='b')


probability=tf.nn.softmax(tf.matmul(x,w)+b)

#构建交叉熵损失函数
loss=tf.reduce_mean(-tf.reduce_sum(y*tf.log(probability),axis=1))
#loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_hat,logits=y_pre,name='loss')

#设置学习率
learn_rate=0.01

#使用一般梯度下降方法的优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
train=optimizer.minimize(loss,name='train')


#tf.argmax(probability,axis=1)  按行取概率最大的数的列索引信息
#tf.argmax(y_data,axis=1)        实际按行取最大的数的列索引信息
#tf.equal 用于判断左右两边的列索引信息是否相同，相同则分类正确
equal=tf.equal(tf.argmax(probability,axis=1),tf.argmax(y,axis=1))
#取均值计算正确率
correct=tf.reduce_mean(tf.cast(equal,tf.float32))

#建立初始化变量init
init=tf.global_variables_initializer()

#创建easy_print方法
def easy_print(p_epoch,p_train_epochs,p_loss,p_accuracy):
    print("迭代次数:{}/{}, 损失值：{},训练集上的准确率：{}".format(p_epoch,p_train_epochs,p_loss,p_accuracy))

#定义总迭代次数
training_epochs=5000
#定义小批量梯度下降的值为10，即使用10条数据迭代一次权重项和偏置参数
batch_size=int(n/10)


with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
    #变量初始化
    sess.run(init)
    print(sess.run(loss,feed_dict={x:x_data[1:10],y:y_data[1:10]}))
    #迭代训练
    for epoch in range(training_epochs):
        #打乱数据集的顺序，返回数据集的索引信息
        #使得每次迭代使用的数据不一致，提高数据的复用
        p_loss=0
        p_correct=0
        index=np.random.permutation(n)
        for i in range(batch_size):
            #获取训练集索引信息
            train_index=index[i*10:(i+1)*10]
            #feed传入参数,并进行模型训练
            sess.run(train,feed_dict={x:x_data[train_index],y:y_data[train_index]})
            #获取损失值,这里计算的是每一个epoch下的batch_size次迭代的平均损失值
            p_loss+=sess.run(loss,feed_dict={x:x_data[train_index],y:y_data[train_index]})/batch_size
            #计算模型的正确率
            p_correct+=sess.run(correct,feed_dict={x:x_data[train_index],y:y_data[train_index]})/batch_size
            easy_print(epoch,training_epochs,p_loss,p_correct)
    result=sess.run([w,b])
    print("最终模型参数为：w={},\nb={}".format(result[0],result[1]))
print("模型训练完成")


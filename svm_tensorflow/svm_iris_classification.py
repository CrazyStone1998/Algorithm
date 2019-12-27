# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from pylab import mpl
from tensorflow.examples.tutorials.mnist import input_data

# 随机种子
np.random.seed(666)
tf.compat.v1.set_random_seed(666)
'''
                                    数据预处理
'''
# iris dataset
iris=datasets.load_iris()
feature_num = 4
x_vals=iris.data # shape(150，4)
y_vals=np.array([1 if y==0 else -1 for y in iris.target]) # shape(150,)

# 划分数据为训练集和测试集 比例 8：2
train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]



# 批训练中批的大小

batch_size = 100
x_data = tf.compat.v1.placeholder(shape=[None, feature_num], dtype=tf.float32)
y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random.normal(shape=[feature_num,1]))
b = tf.Variable(tf.random.normal(shape=[1,1]))

#定义损失函数

model_output=tf.matmul(x_data,W)+b
l2_norm = tf.reduce_sum(tf.square(W))

#软正则化参数

alpha = tf.constant([0.1])

#定义损失函数
classification_term = tf.reduce_mean(tf.maximum(0.,1.-model_output*y_target))
loss = classification_term+alpha*l2_norm

#输出
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target),tf.float32))
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss)

#开始训练
loss_vec = []
train_accuracy = []
test_accuracy = []
with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
   
    for i in range(1000):
       
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
    
        rand_y = np.transpose([y_vals_train[rand_index]])

        sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_accuracy.append(train_acc_temp)
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_accuracy.append(test_acc_temp)
        if (i+1)%100==0:
            print('Step #' + str(i+1),'\nTest_Accuracy = ' + str(test_acc_temp),'\nLoss = '+str(temp_loss))
            print()
        
plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.ylim(0.,1.)
plt.legend(['loss','train_accuracy','test_accuracy'])
plt.savefig('svm_iris_classification.png')
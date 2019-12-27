# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data

# 随机种子
np.random.seed(13)
tf.compat.v1.set_random_seed(13)
'''
                                    数据预处理
'''
# mnist dataset
mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)
feature_num = 784

x_vals_train = mnist.train.images
x_vals_test = mnist.test.images
y_vals_train = np.array([1 if i[0]==1 or i[1]==1 or i[2] == 1 or i[3] == 1 else -1 for i in mnist.train.labels])
y_vals_test = np.array([1 if i[0]==1 or i[1]==1 or i[2] == 1 or i[3] == 1 else -1 for i in mnist.test.labels])


# 批训练中批的大小

batch_size = 500
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
    
    for i in range(4000):
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
            print('Step #' + str(i+1),'\nTrain_Accuracy = '+str(train_acc_temp),'\nTest_Accuracy = ' + str(test_acc_temp),'\nLoss = '+str(temp_loss))
            print()
        
plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.ylim(0.,1.)
plt.legend(['loss','train_accuracy','test_accuracy'])
plt.savefig('svm_mnist_classification.png')
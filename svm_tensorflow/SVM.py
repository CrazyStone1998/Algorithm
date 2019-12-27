# -*- coding: utf-8 -*-

'''
Author: 
        Crazy_Stone
Modify:
        2019-12-25
Project:
        python algorithm experiment
'''

import numpy as np
import pandas as pd 
import matplotlib as mpl
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class SVM:
    
    def __init__(self, dataSet='mnist', kernel='linear', batch_size='100'):
        self._dataSet = dataSet
        self._kernel = kernel
        self._batch_size = batch_size
        self._feature_nums,\
            self.x_vals_train,self.x_vals_test,\
            self.y_vals_train,self.y_vals_test = self._dataSet(self._dataSet)


    def _dataSet_load(self, _dataSet):
        
        if self._dataSet == 'mnist':
            mnist = input_data.read_data_sets('MNIST_DATA/',one_hot=True)
            rows,feature_num = mnist.train.images.shape

            x_vals_train = mnist.train.images # shape(55000,784)
            x_vals_test = mnist.test.images   # shape(10000,784)
            y_vals_train = np.array([1 if i[0]==1 or i[1]==1 or i[2] == 1 or i[3] == 1 else -1 for i in mnist.train.labels])
            y_vals_test = np.array([1 if i[0]==1 or i[1]==1 or i[2] == 1 or i[3] == 1 else -1 for i in mnist.test.labels])

        elif self._dataSet == 'iris':

            iris = datasets.load_iris()
            x_vals = iris.data # shape(150,4)
            y_vals = np.array([1 if y ==1 else -1 for y in iris.target]) # shape(150,)
            rows,feature_num = iris.data.shape

            # 划分数据为训练集和测试集 比例 8：2
            train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8), replace=False)
            test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

            x_vals_train = x_vals[train_indices]
            x_vals_test = x_vals[test_indices]
            y_vals_train = y_vals[train_indices]
            y_vals_test = y_vals[test_indices]

        return feature_num, x_vals_train, x_vals_test, y_vals_train, y_vals_test

    # Gaussian (RBF) kernel
    def _RBF(self, x_data, gamma=-25.0):
        # 创建高斯核函数
        gamma = tf.constant(-50.0)
        dist = tf.reduce_sum(tf.square(x_data), 1)
        dist = tf.reshape(dist, [-1, 1])
        # 实现了(xi-xj)的平方项
        sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
        my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
    
    def _kernel(self, x_data, y_target):
        if self._kernel == 'liner':
            my_kernel = 1
        
        elif self._kernel == 'RBF':
            
            return self._RBF(x_data=x_data)

    
    def _loss(self,):
        pass
    
    def _prediction(self,):
        pass
    
    def _model(self,):
        
        self.x_data = tf.compat.v1.placeholder(shape=[None, feature_num], dtype=tf.float32)
        self.y_target = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
        self.W = tf.Variable(tf.random.normal(shape=[feature_num,1]))
        self.b = tf.Variable(tf.random.normal(shape=[1,1]))

    def fit(self, step=1000):

        #开始训练
        loss_vec = []
        train_accuracy = []
        test_accuracy = []
        self._model()
        with tf.compat.v1.Session() as sess:
            
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            
            for i in range(step):
                rand_index = np.random.choice(len(self.x_vals_train), size=self._batch_size)
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

def main():
    pass


if __name__ == 'main':
    pass


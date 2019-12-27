import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data

# 随机种子
np.random.seed(666)
tf.compat.v1.set_random_seed(666)
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


# 声明批量大小
batch_size = 1000
x_data = tf.placeholder(shape=[None, feature_num], dtype=tf.float32)
# 样本点的数据y为一个1或者-1的数据
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

alpha = tf.Variable(tf.random.normal(shape=[1, batch_size]))

# 创建高斯核函数
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1, 1])
# 实现了(xi-xj)的平方项
sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# 处理对偶问题
# 损失函数对偶问题的第一项
first_term = tf.reduce_sum(alpha)
alpha_vec_cross = tf.matmul(tf.transpose(alpha), alpha)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
# 损失函数对偶问题的第二项
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(alpha_vec_cross, y_target_cross)))
# 第一项加第二项的负数
loss = tf.negative(tf.subtract(first_term, second_term))

# 预测输出
prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), alpha), my_kernel)
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# 创建优化器函数
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# 开始迭代训练
loss_vec = []
train_accuracy = []
test_accuracy = []

with tf.compat.v1.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_test = np.random.choice(len(x_vals_test), size=batch_size)
        
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_accuracy.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test[rand_test], y_target: np.transpose([y_vals_test[rand_test]])})
        test_accuracy.append(test_acc_temp)

        if (i+1)%100==0:
            print('Step #' + str(i+1),'\nTrain_Accuracy = '+str(train_acc_temp),'\nTest_Accuracy = ' + str(test_acc_temp),'\nLoss = '+str(temp_loss))
            print()
        


# plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.ylim(0.,1.)
plt.legend(['train_accuracy','test_accuracy'])
plt.savefig('rbf_svm_mnist_classification.png')

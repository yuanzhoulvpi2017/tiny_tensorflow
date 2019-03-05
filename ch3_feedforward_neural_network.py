import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt 
from random import *
#数据来自kaggle  https://www.kaggle.com/zalando-research/fashionmnist#t10k-images-idx3-ubyte
data_train = pd.read_csv('D:/data/fashion-mnist_train.csv', header=0)
data_train.head()

labels = data_train['label'].values.reshape(1, 60000)
train = data_train.drop('label', axis=1).transpose()
train.shape

train = np.array(train / 255)

#build a model with tensorflow

#modifying labels for the softmax function - one-hot encoding
labels_ = np.zeros((60000, 10))
labels_[np.arange(60000), labels] = 1
labels_ = labels_.transpose()
labels_ = np.array(labels_)
labels_.shape

#the tehsorflow model
n_dim = 784
tf.reset_default_graph()

#number of neurons in the layers
n1 = 5 #number of neruons in layer 1
n2 = 10 #number of neurons in output layer

cost_history = np.empty(shape=[1], dtype=float)
learning_rate = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [10, None])
w1 = tf.Variable(tf.truncated_normal([n1, n_dim], stddev=0.1))
b1 = tf.Variable(tf.zeros([n1, 1]))
w2 = tf.Variable(tf.truncated_normal([n2, n1], stddev=0.1))
b2 = tf.Variable(tf.zeros([n2, 1]))

#let us build our network
Z1 = tf.nn.relu(tf.matmul(w1, X) + b1)
Z2 = tf.nn.relu(tf.matmul(w2, Z1) + b2)
y_ = tf.nn.softmax(Z2, 0)

cost = - tf.reduce_mean(Y * tf.log(y_) + (1 - Y) * tf.log(1-y_))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_epoch = 100  #准确率并不高，可以提高epochs，到5000或者更多
cost_history = []

for epoch in range(train_epoch + 1):
    sess.run(optimizer, feed_dict={X:train, Y:labels_, learning_rate: 0.001})
    cost_ = sess.run(cost, feed_dict={X:train, Y:labels_, learning_rate: 0.001})
    cost_history = np.append(cost_history, cost_)
    if epoch % 10 == 0:
        print('epoch: {}, cost J = {:.6f}'.format(epoch, cost_))

#查看准确率
correct_predictions = tf.equal(tf.argmax(y_, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float')) #转换类型
print("Accuracy: {}".format(accuracy.eval({X:train, Y:labels_, learning_rate:0.001}, session=sess)))

#在训练集上验证
data_dev = pd.read_csv('D:/data/fashion-mnist_test.csv', header=0)
labels_dev = data_dev['label'].values.reshape(1, 10000)

labels_dev_ = np.zeros((10000, 10))
labels_dev_[np.arange(10000), labels_dev] = 1
labels_dev_ = labels_dev_.transpose()

dev = data_dev.drop('label', axis=1).transpose()

#在训练集上查看准确率
correct_predictions = tf.equal(tf.argmax(y_, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float')) #转换类型
print("Accuracy: {}".format(accuracy.eval({X:dev, Y:labels_dev_, learning_rate:0.001}, session=sess)))

#准确率并不高，可以提高epochs，到5000或者更多


##############################################################
#gradient descent Variations
#batch gradient descent
sess = tf.session()
sess.run(tf.global_variables_initializer())
train_epoch = 100

cost_history = []
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
for epoch in range(train_epoch):
    sess.run(optimizer, feed_dict={X:train, Y:labels_, learning_rate:0.01})
    cost_ = sess.run(cost, feed_dict={X:train, Y:labels_, learning_rate:0.01})
    cost_history = np.append(cost_history, cost_)
    if epoch % 5 == 0:
        print('epoch: {}, cost J = {:.6f}'.format(epoch, cost_))

#查看准确率
correct_predictions = tf.equal(tf.argmax(y_, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float')) #转换类型
print("Accuracy: {}".format(accuracy.eval({X:train, Y:labels_, learning_rate:0.01}, session=sess)))

##############################################################
#stochastic gradient descent
cost_history = []
optimizer = tf.train
for epoch in range(train_epoch):
    sess.run(optimizer, feed_dict={X:train, Y:labels_, learning_rate:0.01})
    cost_ = sess.run(cost, feed_dict={X:train, Y:labels_, learning_rate:0.01})
    cost_history = np.append(cost_history, cost_)
    if epoch % 5 == 0:
        print('epoch: {}, cost J = {:.6f}'.format(epoch, cost_))

#查看准确率
correct_predictions = tf.equal(tf.argmax(y_, 0), tf.argmax(Y, 0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float')) #转换类型
print("Accuracy: {}".format(accuracy.eval({X:train, Y:labels_, learning_rate:0.01}, session=sess)))



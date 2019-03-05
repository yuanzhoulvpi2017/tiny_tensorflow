# 在ipython下运行， 非常耗时间
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import random
lst1 = random.sample(range(1, 10**8), 10**7)
lst2 = random.sample(range(1, 10**8), 10**7)
ab = [lst1[i] * lst2[i] for i in range(len(lst1))]

lst1_np = np.array(lst1)
lst2_np = np.array(lst2)
% % timeit
out2 = np.multiply(lst1_np, lst2_np)

#

boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)

print(boston['DESCR'])

n_training_samples = features.shape[0]
n_dim = features.shape[1]
print('The dataset has {} training samples.'.format(n_training_samples))
print('The dataset has {} features.'.format(n_dim))


def normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


features_norm = normalize(features)
train_x = np.transpose(features_norm)
train_y = np.transpose(labels)
train_y.shape
train_x.shape
train_y = train_y.reshape(1, len(train_y))
train_y.shape

# neuron and cost function for linear regression
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])
learning_rate = tf.placeholder(tf.float32, shape=())
W = tf.Variable(tf.ones([n_dim, 1]))
b = tf.Variable(tf.zeros(1))

init = tf.global_variables_initializer()
y_ = tf.matmul(tf.transpose(W), X) + b
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


def run_linear_model(learning_r, training_epochs, train_obs, train_labels, debug=False):
    sess = tf.Session()
    sess.run(init)
    cost_history = np.empty(shape=[0], dtype=float)
    for epoch in range(training_epochs+1):
        sess.run(training_step, feed_dict={
                 X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_ = sess.run(cost, feed_dict={
                         X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)

        if (epoch % 100 == 0) & debug:
            print("Reached epoch {} , cost J = {:.6f}".format(epoch, cost_))
    return sess, cost_history


sess, cost_history = run_linear_model(
    learning_r=0.01, training_epochs=10000, train_obs=train_x, train_labels=train_y, debug=True)



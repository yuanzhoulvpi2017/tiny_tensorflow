import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import load_boston
import sklearn.linear_model as sk 

boston = load_boston()
features = np.array(boston.data)
target = np.array(boston.target)

def normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

feature_norm = normalize(features)
np.random.seed(42)
rnd = np.random.rand(len(feature_norm)) < 0.8

train_x = np.transpose(feature_norm[rnd])
train_y = np.transpose(target[rnd])
dev_x = np.transpose(feature_norm[~rnd])
dev_y = np.transpose(target[~rnd])

train_y = train_y.reshape(1, len(train_y))
dev_y = dev_y.reshape(1, len(dev_y))

def create_layer(X, n, activation):
    ndim = int(X.shape[0])
    stddev = 2.0 / np.sqrt(ndim)
    initialization = tf.truncated_normal((n, ndim), stddev=stddev)
    W = tf.Variable(initialization)
    b = tf.Variable(tf.zeros([n, 1]))
    Z = tf.matmul(W, X) + b
    return activation(Z), W, b 

#the network 
tf.reset_default_graph()

n_dim = 13
n1 = 20
n2 = 20
n3 = 20
n4 = 20
n_ouputs = 1
tf.set_random_seed(42)

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])

learning_rate = tf.placeholder(tf.float32, shape=())

hidden1, w1, b1 = create_layer(X, n1, activation=tf.nn.relu)
hidden2, w2, b2 = create_layer(hidden1, n1, activation=tf.nn.relu)
hidden3, w3, b3 = create_layer(hidden2, n2, activation=tf.nn.relu)
hidden4, w4, b4 = create_layer(hidden3, n3, activation=tf.nn.relu)
y_, w5, b5 = create_layer(hidden4, n4, activation=tf.identity)

cost = tf.reduce_mean(tf.square(y_ - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_train_history = []
cost_dev_history = []

for epoch in range(10000+1):
    sess.run(optimizer, feed_dict={X:train_x, Y:train_y, learning_rate: 0.001})
    cost_train_ = sess.run(cost, feed_dict={X:train_x, Y:train_y, learning_rate: 0.001})
    cost_dev_ = sess.run(cost, feed_dict={X:dev_x, Y:dev_y, learning_rate: 0.001})
    cost_train_history = np.append(cost_train_history, cost_train_)
    cost_dev_history = np.append(cost_dev_history, cost_dev_)

    if epoch % 10 == 0:
        print('epoch: {}, cost J(train) = {:.6f}, cost J(test) = {:.6f}'.format(epoch, cost_train_, cost_dev_))


plt.plot(range(len(cost_train_history)), cost_train_history, label='train')
plt.plot(range(len(cost_dev_history)), cost_dev_history, label='test')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(loc='best')
plt.show()

#w1的各个数值可视化，查看各个数值对应的分布关系
np_w1 = w1.eval(session=sess)
np_w1 = np_w1.reshape(20*13, 1)
np_w1.shape
plt.hist(np_w1, bins=100)
plt.show()


#tensorflow implementation
#下面给w1, w2, w3, w4, w5加上权重。结果效果显著， 不再过拟合

tf.reset_default_graph()

n_dim = 13
n1 = 20
n2 = 20
n3 = 20
n4 = 20
n_ouputs = 1
tf.set_random_seed(42)

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])

learning_rate = tf.placeholder(tf.float32, shape=())

hidden1, w1, b1 = create_layer(X, n1, activation=tf.nn.relu)
hidden2, w2, b2 = create_layer(hidden1, n1, activation=tf.nn.relu)
hidden3, w3, b3 = create_layer(hidden2, n2, activation=tf.nn.relu)
hidden4, w4, b4 = create_layer(hidden3, n3, activation=tf.nn.relu)
y_, w5, b5 = create_layer(hidden4, n4, activation=tf.identity)

lambd = tf.placeholder(tf.float32, shape=())
reg = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5)
cost_mse = tf.reduce_mean(tf.square(y_ - Y))
cost = tf.reduce_mean(cost_mse + lambd * reg)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_train_history = []
cost_dev_history = []
learning_r = 0.001
lambd_val = 0.1

for epoch in range(10000+1):
    sess.run(optimizer, feed_dict={X:train_x, Y:train_y, learning_rate: learning_r, lambd: lambd_val})
    cost_train_ = sess.run(cost, feed_dict={X:train_x, Y:train_y, learning_rate: learning_r, lambd: lambd_val})
    cost_dev_ = sess.run(cost, feed_dict={X:train_x, Y:train_y, learning_rate: learning_r, lambd: lambd_val})
    cost_train_history = np.append(cost_train_history, cost_train_)
    cost_dev_history = np.append(cost_dev_history, cost_dev_)

    if epoch % 10 == 0:
        print('epoch: {}, cost J(train) = {:.6f}, cost J(test) = {:.6f}'.format(epoch, cost_train_, cost_dev_))


plt.plot(range(len(cost_train_history)), cost_train_history, label='train')
plt.plot(range(len(cost_dev_history)), cost_dev_history, label='test')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(loc='best')
plt.show()

#添加dropout

tf.reset_default_graph()

n_dim = 13
n1 = 20
n2 = 20
n3 = 20
n4 = 20
n_ouputs = 1
tf.set_random_seed(42)

X = tf.placeholder(tf.float32, [n_dim, None])
Y = tf.placeholder(tf.float32, [1, None])

learning_rate = tf.placeholder(tf.float32, shape=())
keep_prob = tf.placeholder(tf.float32, shape=())

hidden1, w1, b1 = create_layer(X, n1, activation=tf.nn.relu)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

hidden2, w2, b2 = create_layer(hidden1_drop, n1, activation=tf.nn.relu)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

hidden3, w3, b3 = create_layer(hidden2_drop, n2, activation=tf.nn.relu)
hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

hidden4, w4, b4 = create_layer(hidden3_drop, n3, activation=tf.nn.relu)
hidden4_drop = tf.nn.dropout(hidden4, keep_prob)
y_, w5, b5 = create_layer(hidden4, n4, activation=tf.identity)

cost = tf.reduce_mean(tf.square(y_ - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_train_history = []
cost_dev_history = []

for epoch in range(10000+1):
    sess.run(optimizer, feed_dict={X:train_x, Y:train_y, learning_rate: 0.001, keep_prob:1.0})
    cost_train_ = sess.run(cost, feed_dict={X:train_x, Y:train_y, learning_rate: 0.001, keep_prob:1.0})
    cost_dev_ = sess.run(cost, feed_dict={X:dev_x, Y:dev_y, learning_rate: 0.001, keep_prob:1.0})
    cost_train_history = np.append(cost_train_history, cost_train_)
    cost_dev_history = np.append(cost_dev_history, cost_dev_)

    if epoch % 10 == 0:
        print('epoch: {}, cost J(train) = {:.6f}, cost J(test) = {:.6f}'.format(epoch, cost_train_, cost_dev_))


plt.plot(range(len(cost_train_history)), cost_train_history, label='train')
plt.plot(range(len(cost_dev_history)), cost_dev_history, label='test')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(loc='best')
plt.show()


#从上面不同结果发现，给各个w做约束的时候，效果很明显，不会过拟合，
# 而简单的网络或者加drop的网络就过拟合，其实我没有调参数，所以也不具有代表性，需要去不断地调整参数
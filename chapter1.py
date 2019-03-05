import tensorflow as tf 

#createing and running a computional graph
#computational graph with tf.constant
x1 = tf.constant(1)
x2 = tf.constant(2)
z = tf.add(x1, x2)
sess = tf.Session()
print(sess.run(z))
sess.close()

#computational graph with tf.variable
x1 = tf.Variable(1)
x2 = tf.Variable(2)
z = tf.add(x1, x2)
sess = tf.Session()
print(sess.run(z))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(z))
sess.close()

#computational graph with tf.placeholder
x1 = tf.placeholder(tf.float32, 1)
x2 = tf.placeholder(tf.float32, 1)
z = tf.add(x1, x2)
print(z)
sess = tf.Session()
feed_dict = {x1: [1], x2: [2]}
print(sess.run(z, feed_dict=feed_dict))

x1 = tf.placeholder(tf.float32, [2])
x2 = tf.placeholder(tf.float32, [2])
z = tf.add(x1, x2)
feed_dict = {x1: [1, 5], x2: [1, 1]}
sess = tf.Session()
sess.run(z, feed_dict=feed_dict)

x1 = tf.placeholder(tf.float32, 1)
w1 = tf.placeholder(tf.float32, 1)
x2 = tf.placeholder(tf.float32, 1)
w2 = tf.placeholder(tf.float32, 1)
z1 = tf.multiply(x1, w1)
z2 = tf.multiply(x2, w2)
z3 = tf.add(z1, z2)
feed_dict = {x1: [1], w1: [2], x2: [3], w2: [4]}
sess = tf.Session()
sess.run(z3, feed_dict)


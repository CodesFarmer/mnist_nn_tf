import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
w1 = weight_variable([784, 30])
b1 = bias_variable([30])

#w1 = tf.Variable(tf.random_normal([784, 30]))
#b1 = tf.Variable(tf.zeros([30]))

y1 = tf.nn.sigmoid(tf.matmul(x, w1)+b1)

w2 = weight_variable([30, 10])
b2 = bias_variable([10])

#w2 = tf.Variable(tf.random_normal([30, 10]))
#b2 = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
y = tf.matmul(y1, w2)+b2
#y = tf.matmul(x, w1)+b1

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    #train_step.run(feed_dict={x:batch[0], y_:batch[1]})
    #print(sess.run(cross_entropy, {x:batch[0], y_:batch[1]}))
    _,loss_value = sess.run([train_step,cross_entropy],feed_dict={x:batch[0], y_:batch[1]})
    print("The Loss Value is %g"%loss_value)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

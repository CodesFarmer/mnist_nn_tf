import tensorflow as tf


#The loss function
def loss(y_groundth, y_predicted):
    y_groundth = tf.to_int64(y_groundth)
    wrong_elem = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_groundth, logits=y_predicted)
    cross_entropy = tf.reduce_mean(wrong_elem)
    return cross_entropy

def training(loss, lr):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)
    return train_step

def neuralnetwork(input, c1, c2, fc1, fc2, dpr):

    with tf.name_scope("conv1"):
        initials = tf.truncated_normal(c1,stddev=0.1)
        weights = tf.Variable(initials)
        bias = tf.Variable(tf.zeros(c1[-1]))
        images = tf.reshape(input, [-1,28,28,1])
        h_convl1 = tf.nn.relu(tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding="SAME")+bias)
        h_maxpl1 = tf.nn.max_pool(h_convl1, [1,2,2,1],[1,2,2,1],padding="SAME")

    with tf.name_scope("conv2"):
        initials = tf.truncated_normal(c2, stddev=0.1)
        weights = tf.Variable(initials)
        bias = tf.Variable(tf.zeros(c2[-1]))
        h_convl2 = tf.nn.relu(tf.nn.conv2d(h_maxpl1, weights, [1,1,1,1],padding="SAME")+bias)
        h_maxpl2 = tf.nn.max_pool(h_convl2, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")

    with tf.name_scope("fc1"):
        initials = tf.truncated_normal(fc1,stddev=0.1)
        weights = tf.Variable(initials)
        bias = tf.Variable(tf.zeros(fc1[-1]))
        h_maxpl2_vec = tf.reshape(h_maxpl2, [-1, fc1[0]])
        h_fcl1 = tf.nn.relu(tf.matmul(h_maxpl2_vec, weights)+bias)
        h_fcl1_drop = tf.nn.dropout(h_fcl1, dpr)

    with tf.name_scope("fc2"):
        initials = tf.truncated_normal(fc2, stddev=0.1)
        weights = tf.Variable(initials)
        bias = tf.Variable(tf.zeros(fc2[-1]))
        h_fc2l = tf.matmul(h_fcl1_drop,weights)+bias

    logits = h_fc2l
    return logits

def evaluate_nn(logits, labels):
    corrections = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
    return accuracy
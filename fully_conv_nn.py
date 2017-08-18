import tensorflow as tf
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mnistnn

FLAGS=None

def inputs_placeholder(batch_size):

    pl_input_images = tf.placeholder(tf.float32, shape=[batch_size, 784])
    pl_input_labels = tf.placeholder(tf.int32, shape=[batch_size])
    pl_keep_probs = tf.placeholder(tf.float32)
    return pl_input_images,pl_input_labels,pl_keep_probs

def fill_feed_dict(data_set, pl_input_img, pl_input_l, pl_keep_probs, dpr):

    input_img, input_l = data_set.next_batch(FLAGS.batch_size, False)
    feed_dict = {
        pl_input_img: input_img,
        pl_input_l: input_l,
        pl_keep_probs:dpr
    }
    return feed_dict

def train_network():
    dataset = input_data.read_data_sets('MNIST_data', False)
    with tf.Graph().as_default():
        ph_images, ph_labels, ph_keeprobs = inputs_placeholder(FLAGS.batch_size)
        conv_size = FLAGS.conv1_size[1:-1].split(',')
        conv1_size = [int(conv_size[0]),int(conv_size[1]),int(conv_size[2]),int(conv_size[3])]
        conv_size = FLAGS.conv2_size[1:-1].split(',')
        conv2_size = [int(conv_size[0]),int(conv_size[1]),int(conv_size[2]),int(conv_size[3])]
        fc1_size = [7*7*conv2_size[3], FLAGS.fc1_size]
        fc2_size = [FLAGS.fc1_size, 10]
        logits = mnistnn.neuralnetwork(ph_images, conv1_size,conv2_size,fc1_size,fc2_size,ph_keeprobs)
        cross_entropy = mnistnn.loss(ph_labels, logits)
        train_step = mnistnn.training(cross_entropy, FLAGS.learning_rate)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.max_steps):
            feed_dict = fill_feed_dict(dataset.train, ph_images, ph_labels, ph_keeprobs, FLAGS.dropout_rate)
            _, loss_value = sess.run([train_step,cross_entropy], feed_dict=feed_dict)
            if(step%100==0):
                print("(%d/%d)The loss is %g"%(step, FLAGS.max_steps, loss_value))

        right_samples = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits,ph_labels,1), tf.int32))
        corrected_num = 0
        for iter in range(200):
            print(iter)
            feed_dict = fill_feed_dict(dataset.test, ph_images,ph_labels,ph_keeprobs,1.0)
            corrected_num += sess.run(right_samples, feed_dict)

        print("The accuracy of test set is %g"%(float(corrected_num)/10000.0))

def main(_):

    train_network()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--learning_rate',
        type = float,
        default=0.001,
        help='The learning rate during training...'
    )
    parse.add_argument(
        '--dropout_rate',
        type = float,
        default=0.5,
        help='Dropout the neurons in fully connected layer randomly at this chance...'
    )
    parse.add_argument(
        '--max_steps',
        type = int,
        default = 10000,
        help='The max round for training...'
    )
    parse.add_argument(
        '--conv1_size',
        type = str,
        default = "[3,3,1,64]",
        help='The size of first convolutional layer...'
    )
    parse.add_argument(
        '--conv2_size',
        type = str,
        default="[5,5,64,64]",
        help='The size of second convolutional layer...'
    )
    parse.add_argument(
        '--fc1_size',
        type = int,
        default= 1024,
        help='The size of first fully connected layer...'
    )
    parse.add_argument(
        '--batch_size',
        type = int,
        default=50,
        help = 'The batch size for each round...'
    )

    FLAGS, unparse = parse.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparse)
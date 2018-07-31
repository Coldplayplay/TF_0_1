# _*_ coding:utf-8 _*_
#第二节-卷积神经网络

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from datetime import timedelta
from mnist import MNIST
import time
import math
import os
import prettytensor as pt

#创建一些全局变量
#一些图像参数
img_size = 28
img_channel = 1
img_size_flat = img_size * img_size * img_channel
num_class = 10

#一些模型参数
conv1_filter_size = 5
conv1_filter_num = 16
conv2_filter_size = 5
conv2_filter_num = 36
fc_size = 128

#1.创建新变量的帮助函数
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#2.创建卷积层
def new_conv_layer(input, filter_size, num_filter, stride=1, use_pooling=True):
    num_input_channel = input.get_shape()[-1].value
    shape = [filter_size, filter_size, num_input_channel, num_filter]
    weights = new_weights(shape)
    biases = new_biases(num_filter)
    strides = [1, stride, stride, 1]
    conv = tf.nn.conv2d(input, weights, strides=strides, padding='SAME')
    conv += biases

    if use_pooling:
        conv = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    conv = tf.nn.relu(conv)

    return conv, weights

#3.转换一个层的shape,拉成一个向量
def flatten_layer(conv):
    shape = conv.get_shape()
    #num_feature = shape[1]*shape[2]*shape[3]
    num_feature = shape[1:4].num_elements()
    layer = tf.reshape(conv, [-1, num_feature])
    return layer, num_feature

#4.创建全连接层
def new_fc_layer(layer, output_channnels, use_relu=True):
    layer, num_feature = flatten_layer(layer)
    weights = new_weights([num_feature, output_channnels])
    biases = new_biases(output_channnels)

    out = tf.matmul(layer, weights) + biases

    if use_relu:
        out = tf.nn.relu(out)

    return out, weights


#获取权重对应的tensor
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

#可视化卷积层权重
def plot_conv_weights(session, weights, input_channel=0):
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def model(x, y):
    x_img = tf.reshape(x, [-1, img_size, img_size, img_channel])
    y_cls = tf.argmax(y, axis=1)
    conv1, weights_conv1 = new_conv_layer(x_img, conv1_filter_size, conv1_filter_num)
    conv2, weights_conv2 = new_conv_layer(conv1, conv2_filter_size, conv2_filter_num)
    fc1, weights_fc1 = new_fc_layer(conv2, fc_size)
    fc2, weights_fc2 = new_fc_layer(fc1, num_class, False)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y)
    sum_losses = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(sum_losses)

    y_pred = tf.nn.softmax(fc2)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_cls, y_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy, optimizer, weights_conv1, weights_conv2

def model2(x, y):
    x_img = tf.reshape(x, [-1, img_size, img_size, img_channel])
    y_cls = tf.argmax(y, axis=1)
    x_pretty = pt.wrap(x_img)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=36, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, name='fc1').\
            softmax_classifier(num_class=10, labels=y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    y_pred_cls = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_cls, y_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy, optimizer


def model3(x, y):
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    y_true_cls = tf.argmax(y, axis=1)
    net = tf.layers.conv2d(x_img, name='conv1', filters=16, kernel_size=5,
                           strides=[1, 1], padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2])
    net = tf.layers.conv2d(net, name='conv2', filters=36, kernel_size=5,
                           strides=[1, 1], padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2])
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, name='fc1', units=128, activation=tf.nn.relu)
    net = tf.layers.dense(net, name='fc2', units=10)

    y_pred = tf.nn.softmax(net)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(y_pred_cls, y_true_cls)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y)
    sum_losses = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(sum_losses)

    return acc, opt




data = MNIST()
def run(is_training=True, batch_size=32, iterations=6):
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_class])
    accuracy, optimizer= model3(x, y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_val')

    if is_training:
        for i in range(iterations):
            x_batch, y_batch, y_batch_cls = data.random_batch(batch_size=batch_size)
            feed_dict = {x:x_batch, y:y_batch}
            sess.run(optimizer, feed_dict=feed_dict)

            if i%2 == 0:
                feed_dict = {x: data.x_test, y: data.y_test}
                acc = sess.run(accuracy, feed_dict=feed_dict)
                print('iteration:{0},accuracy:{1}'.format(i, acc))
                saver.save(sess, save_path)

    # weights_conv1 = get_weights_variable('conv1')
    # weights_conv2 = get_weights_variable('conv2')
    # plot_conv_weights(sess, weights_conv1)
    # plot_conv_weights(sess, weights_conv2)





if __name__ == '__main__':
    run()












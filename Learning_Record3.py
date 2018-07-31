# _*_ coding:utf-8 _*_
'''
第三节-封装
a-prettytensor
b-layers
c-keras
'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import prettytensor as pt
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from datetime import timedelta
from mnist import MNIST

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, concatenate
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model

#采用prettytensor实现与Learning_Record2中相同的模型
#对tf.nn.conv2d等底层函数接口实现封装
def plot_conv_weights(w, input_channel=0):
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


def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

data = MNIST()
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
img_shape_full = data.img_shape_full
num_classes = data.num_classes
num_channels = data.num_channels
x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28*1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

#'layer_conv1','layer_conv2','fc1'是层的名字，也叫变量作用域
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')
    return variable
'''
报错
AttributeError: module 'tensorflow.python.ops.variable_scope' 
has no attribute '_VARSCOPE_KEY'
'''
def prettytensor_model(x, y):
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    x_pretty = pt.wrap(x_img)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred, loss = x_pretty.conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=36, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, name='fc1').\
            softmax_classifier(num_class=10, labels=y)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

def layers_model(x, y):
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    y_true_cls = tf.argmax(y, axis=1)
    net = tf.layers.conv2d(x_img, name='conv1', filters=16, kernel_size=5,
                           strides=[1,1], padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2], strides=[2,2])
    net = tf.layers.conv2d(net, name='conv2', filters=36, kernel_size=5,
                           strides=[1,1], padding='SAME', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2], strides=[2,2])
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

    return acc, opt, y_pred

def keras_sequential_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    model.add(Reshape(img_shape_full))
    model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def keras_functional_model():
    inputs = Input(shape=(img_size_flat,))
    net = inputs
    net = Reshape(img_shape_full)(net)
    net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)
    net1 = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2')(net)
    net1 = MaxPooling2D(pool_size=2, strides=2)(net1)

    net2 = Conv2D(kernel_size=3, strides=1, filters=12, padding='same',
                  activation='relu', name='layer_conv3')(net1)
    net2 = MaxPooling2D(pool_size=2, strides=2)(net2)

    #net1 = Flatten()(net1)
    net2 = Flatten()(net1)
    #net = concatenate([net1, net2])

    net = Dense(128, activation='relu')(net2)
    net = Dense(num_classes, activation='softmax')(net)
    outputs = net

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model

'''
#model = keras_sequential_model()
model = keras_functional_model()

model.summary()  #列表，所有的层

#获得具体的层以及模型变量
input = model.layers[0]
conv1 = model.layers[2]
conv2 = model.layers[10]
weight1 = conv1.get_weights()[0]
weight2 = conv2.get_weights()[0]
print(weight1.shape)
print(weight2.shape)
plot_conv_weights(weight1)
plot_conv_weights(weight2)


#某些层的输出
#method1
image1 = data.x_test[0]
from tensorflow.python.keras import backend as K
output_conv1 = K.function(inputs=[input.input],
                          outputs=[conv1.output])
layer_output1 = output_conv1([[image1]])[0]

#method2
output_conv2 = Model(inputs=input.input, outputs=conv2.output)
layer_output2 = output_conv2.predict(np.array([image1]))

print(layer_output2.shape)
plot_conv_output(layer_output2)


#train
model.fit(x=data.x_train[:2000],
          y=data.y_train[:2000],
          epochs=1, batch_size=128)

#test
result = model.evaluate(x=data.x_test[:5],
                        y=data.y_test[:5])
for name, value in zip(model.metrics_names, result):
    print(name, value)

#predict
images = data.x_test[0:9]
y_pred= model.predict(x=images)
print(np.argmax(y_pred, axis=1))

#模型的保存和恢复
path_model = 'model.keras2'
model.save(path_model)
del model
model = load_model(path_model)
y_pred = model.predict(x=data.x_test[:9])
print(np.argmax(y_pred, axis=1))

'''



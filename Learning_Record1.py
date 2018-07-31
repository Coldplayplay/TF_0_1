# _*_ coding:utf-8 _*_

#第一节-线性分类器
#从MNIST入门，完整地走一遍手写字符的识别
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mnist import MNIST

img_size = 28
img_size_flat = 28*28
img_shape = (28, 28)
num_classes = 10


'''
一个TensorFlow图由下面几个部分组成，后面会详细描述：(5部分)
 -占位符变量（Placeholder）用来改变图的输入。
 -模型变量（Model）将会被优化，使得模型表现得更好。
 -模型本质上就是一些数学函数，它根据Placeholder和模型的输入变量来计算一些输出。
 -一个cost度量用来指导变量的优化。
 -一个优化策略会更新模型的变量。
'''

''' graph开始'''

x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat], name='input')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='label')
y_true_cls = tf.placeholder(dtype=tf.int64, shape=[None], name='class')

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

y_logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(y_logits)
y_pred_cls = tf.argmax(y_pred, 1)
correct_prediction = tf.equal(y_true_cls, y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_true)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=y_true_cls)
sum_losses = tf.reduce_mean(loss)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(sum_losses)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(sum_losses)
'''graph结束'''

data = MNIST(data_dir='data/MNIST/')
feed_dict_test = {x:data.x_test, y_true:data.y_test, y_true_cls:data.y_test_cls}

def print_accuracy(sess):
    feed_dict = feed_dict_test
    acc = sess.run(accuracy, feed_dict=feed_dict)
    print('accuracy on the test-set: {0:.1%}'.format(acc))
def print_confusion_matrix(session):
    # Get the true classifications for the test-set.
    cls_true = data.y_test_cls
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    # Print the confusion matrix as text.
    print(cm)
    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
def plot_images(images, cls_true, cls_pred=None):
    assert len(images)==len(cls_true)==9
    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = 'True:{0}'.format(cls_true[i])
        else:
            xlabel = 'True:{0}, Pred:{1}'.format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
def plot_example_errors(sess):
    correct, pred_cls = sess.run([correct_prediction, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct==False)
    incorrect_images = data.x_test[incorrect]
    incorrect_pred = pred_cls[incorrect]
    correct_pred = data.y_test_cls[incorrect]
    plot_images(incorrect_images[0:9], correct_pred[0:9], incorrect_pred[0:9])
    # logits, pred, correct, pred_cls = sess.run([y_logits, y_pred, correct_prediction, y_pred_cls],
    #                                            feed_dict=feed_dict_test)
    # incorrect = (correct==False)
    # incorrect_logits = logits[incorrect]
    # incorrect_pred = pred[incorrect]
    # print(incorrect_logits[0:3])
    # print(incorrect_pred[0:3])
def plot_weights(sess):
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
def run(sess, iterations, batch_size=100):
    for i in range(iterations):
        x_batch, y_batch, y_batch_cls = data.random_batch(batch_size=batch_size)
        feed_dict = {x:x_batch, y_true:y_batch, y_true_cls:y_batch_cls}
        sess.run(optimizer, feed_dict=feed_dict)

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    run(sess, 990)
    print_accuracy(sess)
    plot_example_errors(sess)
    plot_weights(sess)
    print_confusion_matrix(sess)



# _*_ coding:utf-8 _*_
'''
第五节-集成学习
这里采用的集成学习的形式叫 Bootstrap Aggregating（bagging）
是model average的一种特殊形式，它常用来避免过拟合
1994年提出，随机生成训练集，结合每个网络的分类结果来提高分类性能。
神经网络ensemble的作用有点随机，可能无法提供一个提升性能的可靠方式（和单独神经网络性能相比）
你认为集成学习值得更多的研究吗，或者宁可专注于提升单个神经网络的性能？
[bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
结合第四节学习的Saver()来保存和恢复神经网络中的变量,进行练习
分别使用tf.layers和keras两种形式完成集成学习
'''
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.models import load_model
from mnist import MNIST
from Learning_Record3 import layers_model, keras_functional_model

data = MNIST()
img_size = data.img_size                #28
img_size_flat = data.img_size_flat      #28*28*1=784
img_shape = data.img_shape              #(28, 28)
img_shape_full = data.img_shape_full    #(28, 28, 1)
num_classes = data.num_classes          #10
num_channels = data.num_channels        #1

train_val_x_total = np.concatenate([data.x_train, data.x_val], axis=0)
train_val_y_total = np.concatenate([data.y_train, data.y_val], axis=0)
total_num = len(train_val_y_total)
train_num = 55000

def random_train_set():
    idx = np.random.permutation(total_num)
    train_x_set = train_val_x_total[idx[:train_num]]
    train_y_set = train_val_y_total[idx[:train_num]]
    return train_x_set, train_y_set

def random_batch(images, labels, batch_size):
    total = len(images)
    idx = np.random.choice(total, batch_size, replace=False)
    train_x_batch = images[idx]
    train_y_batch = labels[idx]
    return train_x_batch, train_y_batch


#1.使用tf.layers
def use_layers_train(num_net=2, iters=1, batch_size=32):

    for i in range(num_net):
        sess.run(tf.global_variables_initializer())
        x_set, y_set = random_train_set()
        save_path = save_dir + 'net' + str(i)

        for j in range(iters):
            x_batch, y_batch = random_batch(x_set, y_set, batch_size)
            feed_dict = {x:x_batch, y:y_batch}
            sess.run(opt, feed_dict=feed_dict)

        print('net{} trained over.'.format(i))
        print('model saved in {}'.format(save_path))
        saver.save(sess, save_path)

def use_layers_eval(num_net=2):

    feed_dict = {x:data.x_test, y:data.y_test}
    acc_list = []
    pred_list = []

    for i in range(num_net):
        save_path = save_dir + 'net' + str(i)
        saver.restore(sess, save_path)
        accuracy, y_prediction = sess.run([acc, y_pred], feed_dict=feed_dict)

        print('net:{0} , accuracy:{1}'.format(i, accuracy))
        acc_list.append(accuracy)
        pred_list.append(y_prediction)

    acc_array = np.array(acc_list)
    pred_array = np.array(pred_list)
    ensemble_pred = np.mean(pred_array, axis=0)
    ensemble_pred_cls = np.argmax(ensemble_pred, axis=1)
    correct = np.equal(ensemble_pred_cls, data.y_test_cls)
    acc_ensemble = np.sum(correct)/len(correct)/1.0

    print('best net:{0}     accuracy:{1}'.format(np.argmax(acc_array), np.amax(acc_array)))
    print('ensemble accuracy:{}'.format(acc_ensemble))


#2.采用keras

def use_keras_train(num_net=5, iters=2, batch_size=32):
    for i in range(num_net):
        print('net{0} trains...'.format(i))
        model = keras_functional_model()
        x_set, y_set = random_train_set()
        model.fit(x=x_set,
                  y=y_set,
                  epochs=iters, batch_size=batch_size)

        path_model = 'model.keras'+ str(i)
        model.save(path_model)
        del model


def use_keras_eval(num_net=5):
    acc_list = []
    pred_list = []
    for i in range(num_net):
        path_model = 'model.keras'+ str(i)
        model = load_model(path_model)
        result= model.evaluate(x=data.x_test,
                            y=data.y_test)

        y_pred = model.predict(x=data.x_test)

        acc_list.append(result[1])
        pred_list.append(y_pred)

    acc_array = np.array(acc_list)
    pred_array = np.array(pred_list)
    ensemble_pred = np.mean(pred_array, axis=0)
    ensemble_pred_cls = np.argmax(ensemble_pred, axis=1)
    correct = np.equal(ensemble_pred_cls, data.y_test_cls)
    acc_ensemble = np.sum(correct) / len(correct) / 1.0

    print('best net:{0}, accuracy:{1}'.format(np.argmax(acc_array), np.amax(acc_array)))
    print('ensemble accuracy:{}'.format(acc_ensemble))

if __name__ == '__main__':
    # method-1 tf.layers
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat])
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
    acc, opt, y_pred = layers_model(x, y)
    sess = tf.Session()
    saver = tf.train.Saver()
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    #use_layers_train(num_net=5, iters=1000, batch_size=32)
    #use_layers_eval(num_net=5)


    #method-2  keras
    #use_keras_train(num_net=5, iters=4)
    use_keras_eval(num_net=1)









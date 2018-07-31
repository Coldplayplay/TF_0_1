# _*_ coding:utf-8 _*_
'''
第七节
直接调用预训练模型

'''
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os

# Functions and classes for loading and using the Inception model.
import inception

def classify(image_path):
    #image = cv2.imread(image_path)
    #cv2.imshow(image_path.split('/')[-1].split('.')[0], image)

    # image = mpimg.imread(image_path)
    # plt.imshow(image)
    # plt.show()

    pred = model.classify(image_path=image_path)
    print(image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)


inception.maybe_download()

model = inception.Inception()

image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)
classify(image_path="images/parrot.jpg")
classify(image_path="images/parrot_cropped1.jpg")
classify(image_path="images/parrot_cropped2.jpg")
classify(image_path="images/parrot_cropped3.jpg")
classify(image_path="images/parrot_padded.jpg")
classify(image_path="images/elon_musk.jpg")
classify(image_path="images/willy_wonka_old.jpg")
classify(image_path="images/willy_wonka_new.jpg")

#cv2.waitKey()
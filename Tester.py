# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:43:31 2021

@author: BRA6COB
"""
import tensorflow as tf
import glob
import cv2
import numpy as np
from tensorflow.contrib.layers import flatten

def conv2d(x, W, b, strides=1,pad='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=pad) #padding='SAME'
    x = tf.nn.bias_add(x, b)
    return x #tf.nn.relu(x)

def maxpool2d(x, k=2,s=1,pad='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=pad) ##padding='SAME'

# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    n_classes = 43
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 6],mu,sigma)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16],mu,sigma)),
        'wd1': tf.Variable(tf.random_normal([400, 120],mu,sigma)),
        'wd2': tf.Variable(tf.random_normal([120, 84],mu,sigma)),
        'out': tf.Variable(tf.random_normal([84, n_classes],mu,sigma))}

    biases = {
        'bc1': tf.Variable(tf.random_normal([6],mu,sigma)),
        'bc2': tf.Variable(tf.random_normal([16],mu,sigma)),
        'bd1': tf.Variable(tf.random_normal([120],mu,sigma)),
        'bd2': tf.Variable(tf.random_normal([84],mu,sigma)),
        'out': tf.Variable(tf.random_normal([n_classes],mu,sigma))}
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    layer1 = conv2d(x, weights['wc1'], biases['bc1'],1,'VALID')
    # TODO: Activation.
    conv1 = tf.nn.relu(layer1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1,2,2,'VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    layer2 = conv2d(conv1, weights['wc2'], biases['bc2'],1,'VALID')
    # TODO: Activation.
    conv2 = tf.nn.relu(layer2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2,2,2,'VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    conv2_flat = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(conv2_flat, weights['wd1']), biases['bd1'])
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return logits

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    images = glob.glob('Traffic_Sign_Web/*.jpg')
    for idx, fname in enumerate(images):
        img_idx = cv2.imread(fname)
        print(img_idx.shape)
        resized_image_idx = cv2.resize(img_idx, (32,32), interpolation= cv2.INTER_AREA)
        print(resized_image_idx.shape)
        #preprocess the image:
        R = resized_image_idx[:,:,0]
        R_norm = normalize_grayscale(R)
        G = resized_image_idx[:,:,1]
        G_norm = normalize_grayscale(G)
        B = resized_image_idx[:,:,2]
        B_norm = normalize_grayscale(B)
        image_norm = np.dstack((R_norm,G_norm,B_norm))
        #X_norm_set.append(image_norm)
        #image_normalized = np.float32(X_norm_set)
        pred = LeNet(image_norm)
        output = sess.run(pred)
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import cv2
from tensorflow.contrib.layers import flatten
import glob

# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
    """
    image_data_mean_centred = image_data - np.mean(image_data)
    image_data_normalized = image_data_mean_centred / np.std(image_data_mean_centred)
    return image_data_normalized

#define the pipeline
def image_preprocess(X_set):
    X_gray_set = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(len(X_set)):
        """
        R = X_set[i][:,:,0]
        R = clahe.apply(R)
        R_norm = normalize_grayscale(R)
        G = X_set[i][:,:,1]
        G = clahe.apply(G)
        G_norm = normalize_grayscale(G)
        B = X_set[i][:,:,2]
        B = clahe.apply(B)
        B_norm = normalize_grayscale(B)
        image_norm = np.dstack((R_norm,G_norm,B_norm))
        """
        X_reshaped = cv2.resize(X_set[i], (32,32), interpolation= cv2.INTER_AREA)
        gray = cv2.cvtColor(X_reshaped, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        image_norm = normalize_grayscale(gray)
        img_expanded = image_norm[:, :, np.newaxis]
        X_gray_set.append(img_expanded)

    image_normalized = np.float32(X_gray_set)
    return image_normalized

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            
def conv2d(x, W, b, strides=1,pad='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=pad) #padding='SAME'
    x = tf.nn.bias_add(x, b)
    return x #tf.nn.relu(x)

def maxpool2d(x, k=2,s=1,pad='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=pad) ##padding='SAME'

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    n_classes = 43
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6],mu,sigma)),
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

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[-1].shape

# TODO: How many unique classes/labels there are in the dataset.
#n_classes = y_train[np.argmax(y_train)]
n_classes = np.max(y_train)+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

"""
length = len(X_train)
f,ax = plt.subplots(10, 10, figsize=(20,10))
for i in range(10):
    for j in range(10):
        ax[i,j].imshow(X_train[(i+1)*(j+1)])
        plt.imshow(X_train[i])


class_count = []
for i in range(n_classes):
    count = 0
    for j in range(len(y_train)):
        if (y_train[j] == i):
            count = count + 1
    class_count.append(count)
print(class_count)
"""
EPOCHS = 10
BATCH_SIZE = 128
print(type(X_train))

#shuffle the datasets
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

#process the image sets using pipeline
X_train = image_preprocess(X_train)
X_valid = image_preprocess(X_valid)
X_test = image_preprocess(X_test)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

X_norm_set = []
Y_set = np.int32([0,13,26,28,31,23,25,24,40])
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    images = sorted(glob.glob('./Traffic_Sign_Web/German_Traffic_Sign_*.jpg'))
    for idx, fname in enumerate(images):
        img_idx = cv2.imread(fname)
        print(fname)
        print(idx)
        print(img_idx.shape)
        image_norm = image_preprocess(img_idx)
        """
        resized_image_idx = cv2.resize(img_idx, (32,32), interpolation= cv2.INTER_AREA)
        print(resized_image_idx.shape)
        #preprocess the image:
        R = resized_image_idx[:,:,0]
        R = clahe.apply(R)
        R_norm = normalize_grayscale(R)
        G = resized_image_idx[:,:,1]
        G = clahe.apply(G)
        G_norm = normalize_grayscale(G)
        B = resized_image_idx[:,:,2]
        B = clahe.apply(B)
        B_norm = normalize_grayscale(B)
        image_norm = np.float32(np.dstack((R_norm,G_norm,B_norm)))
        X_norm_set.append(image_norm)
        """
        plt.imsave('Output_images/web_image_processed_'+str(idx)+'.png',image_norm)
        X_norm_set.append(image_norm)
    
    outside_test_accuracy = evaluate(X_norm_set, Y_set)
    print("Outside Test Accuracy = {:.3f}".format(outside_test_accuracy))
    pred = sess.run(tf.nn.softmax(logits),feed_dict={x: X_norm_set})
    print(pred)
    print(pred.shape)
    prediction = []
    for i in range(len(pred)):
        prediction.append(np.argmax(pred[i]))
        print(prediction[i])
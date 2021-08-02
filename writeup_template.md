# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_images/random_train_images.png "Ramdom Train Images Visualization"
[image2]: ./Output_images/random_train_images_distribution.png "Bar chart distribution of Train images"
[image3]: ./Output_images/random_validation_images.png "Ramdom validation Images Visualization"
[image4]: ./Output_images/random_validation_images_distribution.png "Bar chart distribution of validation images"
[image5]: ./Output_images/random_test_images.png "Ramdom Test Images Visualization"
[image6]: ./Output_images/random_test_images_distribution.png "Bar chart distribution of Test images"
[image7]: ./Output_images/random_web_images.png "Ten images from Web"
[image8]: ./Output_images/FeatureMaps.png "Feature Maps visualization for a Network layer"
[image9]: ./Output_images/web_image_processed_0.png "web image 0 processed with image processing pipeline"
[image10]: ./Output_images/web_image_processed_1.png "web image 1 processed with image processing pipeline"
[image11]: ./Output_images/web_image_processed_2.png "web image 2 processed with image processing pipeline"
[image12]: ./Output_images/web_image_processed_3.png "web image 3 processed with image processing pipeline"
[image13]: ./Output_images/web_image_processed_4.png "web image 4 processed with image processing pipeline"
[image14]: ./Output_images/web_image_processed_5.png "web image 5 processed with image processing pipeline"
[image15]: ./Output_images/web_image_processed_6.png "web image 6 processed with image processing pipeline"
[image16]: ./Output_images/web_image_processed_7.png "web image 7 processed with image processing pipeline"
[image17]: ./Output_images/web_image_processed_8.png "web image 8 processed with image processing pipeline"
[image18]: ./Output_images/web_image_processed_9.png "web image 9 processed with image processing pipeline"
[image19]: ./Output_images/random_batch_image_before_processing.png "A random batch image before image processing pipeline was applied"
[image20]: ./Output_images/random_batch_image_after_processing.png "A random batch image after image processing pipeline was applied"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/balajirajan87/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

First i import the pickle files "train.p", "test.p", "valid.p", and extract their contents.I extract the "Features" and "Labels" datas from "train", "valid" and "test" variables, and store them in X___, y___ variables. 

I see that the individual images are shaped (32,32,3) and this is known from : X_train[n].shape
I see that Training Data set has 34799 samples
I see that Validation Data set has 4410 samples
I see that Training Data set has 12630 samples

I see that there are 43 classes and this is obtained by: np.max(y_train)+1

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First we take the Training set and we take a random 100 images and plot them as shown below:
![alt text][image1]
I have also tried to create the visualization for the Distribution of the 43 classes in the Train Dataset. please refer the image below: we can see that clasess: 1,2,4,5 10,12,13 are ditributed more than the others, and also we can see that the classes 20 to 43 have generally lower distribution than the others.
![alt text][image2]
similarly we take the Validation and test data sets and try to visualize them in  the similar way.
Random 100 images of Validation set:
![alt text][image3]
Distribution of Validation set:
![alt text][image4]
Random 100 images of Test set:
![alt text][image5]
Distribution of Test set:
![alt text][image6]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The Image pre-processing pipeline consists of below steps:

1. Image resizing to 32x32.
2. converstion to gray scale.
3. applying the clahe (contrast adaptive histogram equalization)
4. Normalization of the image to have 0 mean and 1 standard deviation.

I defined two pipelines for image processing. One for batch images, and one for single images from web.
i chose to first start with Image resizing just in case the input images are of sizes other than 32x32x3. This i do with cv2.resize as below:
```
X_reshaped = cv2.resize(X_set[i], (32,32), interpolation= cv2.INTER_AREA)
```
then i convert to gray scale. Note:- having the color image-sets for training also proved to be effective , resultinig in greater validation accuracies. But there i had to individually extract R,G, and B Color spaces and apply Histogram equalizations and Normalization to each of the color spaces. i can approach in either way and eaither way proved effective. 

then i applied the adaptive histogram equalizations because this could help to adjust the contrast when applied to the real time moving images, and also when looking into some of the train data sets i find that some are with low illuminance and the contrast level is very low. so i felt that applying the clache method (contrast adaptive hist) would do the needfull of brightening up each and every images before passing to the neural networks.

Then i applied the Image normalization where i centre the image to the origin by subtracting each and every pixel with the mean value as below
```
image_data_mean_centred = image_data - np.mean(image_data)
```
Then i divided the zero centred image by the standard deviation of the image, as below. in this way my image pixels would be in the range of -1 to +1 with 0 mean.
```
image_data_normalized = image_data_mean_centred / np.std(image_data_mean_centred)
```
The reason why i normalized the image is i wanted the image data to be evenly spread out and i wanted to remove any uneveness that could potentially mislead the neural network. 
I initially trained the neural network without image normalization and the network performed very poorly. 
After image / fataset normalization the network performance in learning the images was drastically improved.

below you can find one random image from the dataset before image preprocessing:
![alt text][image19]
And below you can find the same random image from the dataset after image preprocessing:
![alt text][image20]
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Dropout				|80% probablity to keep the units				|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Dropout				|80% probablity to keep the units				|
| Final Layer			| Logits calc, inputs 84 outputs 43				|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128, and Epoch count as 100, and a Learn rate of 0.001.
first we define the tensorflow placeholders for x and y variables.
We then call the LeNet function with x as the argument, wich is essentially teh above mentioned convolution steps, and then store the result to the variable 'logits'.
Then we calculate the softmax cross entropy with the function call: 'tf.nn.softmax_cross_entropy_with_logits'.
The above mentioned cross entropy has to be reduced and we do that with Adam_optimizer (code below)
```
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```
Next we start a Tensor flow session and first we initialize the tf variables
for 100 times as defined by the epoch count, we shuffle the train data sets.
Next we split the Train data sets into batches of 128 datasets, as per the below code section:

```
for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
```
Now we call the training_operation via tf session, and feed the batch split up train data sets, and this data flows from training_operation variable upto LeNet(x) function where the CNN is being called.

eventhough i intially used an epoch count of 100 i dynamically tried to optimize the epoch count base on the gradient of Validation accuracy. so when the vaidation accuracy did not change much and when the validation accuracy is greater than at least 0.93 i.e, 93% then i call the loop break that exits the for loop that is used to run the tensorflow session. code for reference:
```
acc_grad = validation_accuracy - validation_accuracy_prev
        acc_grad_filt = acc_grad_filt - fact * (acc_grad_filt - acc_grad)
        validation_accuracy_prev = validation_accuracy
        print("accuracy_gradient = {:.3f}".format(acc_grad_filt))
        if ((acc_grad_filt < 0.005) & (validation_accuracy > 0.93)):
            break
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.0%
* validation set accuracy of 93.7%
* test set accuracy of 92.3%

Before feeding the data to the CNN, both the Training and Validation data sets were well shuffled and also same image preprocessing methods as described in the previous session were applied to all the data sets. 

Added to that i designed the Neural network as below:
The first layer is a convolutional layer that has 5x5 weights and 6 biases that converts the 32x32x3 image to a 28x28x6 image.
Next we apply a ReLU layer to nonLinearlize the layer, and then we apply maxpooling with filter size 2 and stride of 2 that converts the 28x28x6 image to a 14x14x6 image.
necxt we repeat the above mentioned steps (convolution, followed by ReLU followed by MaxPool) to convert the 14x14x6 image to a 5x5x16 image.
Next we flatten the layer and convert to layer of 400 units. 
to the flattened layer we apply activation (y = mx+b), followed by ReLU followed by maxpool to convert a 400 unit layer to a 120 unit layer.
in a similar fashion we apply activation followed by ReLu followed by Maxpool once again to 120 unit layer to convert them to 84 unit layer.
Then finally we apply the last activation to convert the 84 unit layer to a 43 unit layer which we call as Logits and we output the Logits.

below is the code for that:

```
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

    layer1 = conv2d(x, weights['wc1'], biases['bc1'],1,'VALID')
    conv1 = tf.nn.relu(layer1)
    conv1 = maxpool2d(conv1,2,2,'VALID')
    layer2 = conv2d(conv1, weights['wc2'], biases['bc2'],1,'VALID')
    conv2 = tf.nn.relu(layer2)
    conv2 = maxpool2d(conv2,2,2,'VALID')
    conv2_flat = flatten(conv2)
    fc1 = tf.add(tf.matmul(conv2_flat, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
```

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some 10 German traffic signs that I found on the web:

![alt text][image7]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

the first 9 signs were correctly classified and these images were taken from : The German Traffic Sign Recognition Benchmark (GTSRB) contains 43 classes of traffic signs. [source:](https://paperswithcode.com/dataset/gtsrb) 

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit(20kph)	| Speed Limit(20kph)							| 
| Yield Sign   			| Yield Sign									|
| Traffic Lights		| Traffic Lights								|
| Children crossing		| Children crossing				 				|
| Wild Animals crossing	| Wild Animals crossing							|
| Slipery road			| Slipery road									|
| Road work				| Road work										|
| Road narrows to the right	| Road narrows to the right					|
| Roundabout			| Roundabout									|


The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%
For the first image, the model is relatively sure that this is a 20 kph speed limit sign (probability of 0.9), and the image does contain a 20 kph speed limit sign. The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9996        			| 20kph speed limit								| 
| .000393  				| 70kph speed limit 							|
| negligible			| 120kph speed limit							|

For the second image ... 
the model is relatively sure that this was a Yield sign (probability of 1.0), and the image does contain a yield sign. The top 3 soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Yield Sign									| 
| 4.99382377e-19		| turn left ahead 								|
| 1.03083042e-23		| stop sign										|

The model had similar performance on all the nine images taken from [source:](https://paperswithcode.com/dataset/gtsrb)

-------------------------------------------------------------------------------------------------------------------------------

But the model performed poorly on the next 4 images taken from other sources, which had different viewing angles, some had different lighting conditions, etc.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of 30 kph limit	| End of speed limit (80km/h)					| 
| Danger curve to left  | Priority Road									|
| Beware of ice/snow	| slippery road									|
| Danger curve to right | Bicycle crossing				 				|

some detailed discussion below:-
The first image: End of Speed Limit (30 Kph) was actually not present in the training dataset itself, hence the network had difficulty in classifying the image. Maybe the network had looked into the cross lines on the 30 symbol and hence it had predicted that it was keep right ??

also the softmax probablities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.62366462e-01        | End of speed limit (80km/h)					| 
| 2.22119782e-02		| End of no passing								|
| 1.30268652e-02		| Speed limit (20km/h)							|

The second image was a danger curve to the left, but the model predicted that it would be: Priority work. Maybe this was because of the background which has lots of yellow color in that. Maybe the model also predicted go straight or right because of the inherent structure of the symbol ?
also the Softmax predictions were:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.41910362e-01        | Priority Road									| 
| 1.47738934e-01		| Go Straight or Right							|
| 1.30268652e-02		| Road work										|

The third image was "beware of ice or snow", but the model predicted that it would be: Slippery road. Maybe this was because the image contained the traffic sign plates stacked on top of one another, and this confused the model to predict it to slippery surface ? Maybe the network also predicted the sign to be children crossing / and dangerous curve to the left due to the inherent structure / the layout of the image symbol ?? 
also the Softmax predictions were:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.48650301e-01		| Slippery road									| 
| 2.50401143e-02		| Children crossing								|
| 1.62659902e-02		| Danger curve to left							|

The fourth image was "Dangerous curve to the right", but the model predicted that it would be: "Bicycle crossing". Maybe this was because of the inherant structure of the symbol. it also predicted to be children crossing as the next highest probable sign. 
the Softmax predictions were:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.79272795e-01		| Bicycle crossing								| 
| 1.27516195e-01		| Children crossing								|
| 7.19555095e-02		| Road work										|

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Note:- softmax discussion for both good performance and bad performance images has been discussed in detail in the previous section.
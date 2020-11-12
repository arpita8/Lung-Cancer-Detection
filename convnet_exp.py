# This script is based on a kernel found on kaggle
# The author of this tutorial is SentDex
# The link to this tutorial is here:
# https://www.kaggle.com/sentdex/data-science-bowl-2017/
# first-pass-through-data-w-3d-convnet
# This is a change in the code

#imports
import dicom  # for reading dicom files
import os  # for doing directory operations
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import tensorflow as tf
import csv
import pickle
from sklearn.metrics import log_loss

# These are the dimensions of our image
# The IMG_SIZE_PX represents a 50x50 image
# the slice count is the number of slices we will combine
# to make a chunk during pre-processing.
IMG_SIZE_PX = 50
SLICE_COUNT = 20

#classifies to 0 and 1
n_classes = 2

##### To understand what it means ####
batch_size = 10
IMG_PX_SIZE = 150
HM_SLICES = 20
keep_rate = 0.9

#### up to here ####

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# This defines the parameters of our model.
# the inputs to tf.nn.conv3d are listed as follows:
# input: A tensor, which can be defined as any numerical value data type. 
#        Its Shape is [batch, in_depth, in_height, in_width, in_channels]
# Filter: A tensor, must be the same datatype as the input, 
#         [filter_depth,filter_height, filter_width, in_channels, out_channlels]
# strides: A list of ints that is length >=5. 1-D tensor of length 5
#          of the sliding window for each dimension of the input. 
#          The stride controls the way the filter convolves around the input. 
#          If the stride is 1 then the filter shifts one unit at a time.
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    # size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 5, 32, 64])),
               #                                  64 features
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x, hm_epochs=20, learning_rate=1e-3):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		successful_runs = 0
		total_runs = 0

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for data in train_data:
				total_runs += 1
				try:
					X = data[0]
					Y = data[1]
					_, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
					epoch_loss += c
					successful_runs += 1
				except Exception as e:
					# I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
					# input tensor. Not sure why, will have to look into it. Guessing it's
					# one of the depths that doesn't come to 20.
					pass
					# print(str(e))

			print('Epoch', epoch + 1, 'completed out of',
				  hm_epochs, 'loss:', epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], 
											y: [i[1] for i in validation_data]}))

		print('Done. Finishing accuracy:')
		final_accuracy = accuracy.eval(
			{x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]})
		print('Accuracy:', final_accuracy)
		print(total_runs)
		print('fitment percent:', successful_runs / total_runs)
		#predict = tf.argmax(prediction, 1)
		test_pred = (prediction.eval({x: [i[0] for i in testing]}))
		print(total_runs)
		print('fitment percent:', successful_runs / total_runs)
	
	return test_pred, final_accuracy

##################################
#       Getting values	         #
##################################

csv_file = 'testingValues.csv'

field_names = ['Epochs number', 'Learning rate', 'accuracy', 'log loss']
'''
with open(csv_file, 'w') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=field_names, lineterminator='\n')
    writer.writeheader()	
'''
solution = pd.read_csv("stage1_sample_submission.csv", index_col='id')
ground_truth = pd.read_csv("stage1_sol.csv", index_col = 'id')


file_path = "stage1_labels.csv"
labels = pd.read_csv(file_path, index_col=0)

training = np.load('traindata-50-50-20.npy')
testing = np.load('testdata-50-50-20.npy')

# If you are working with the basic sample data, use maybe 2 instead of
# 100 here... you don't have enough data to really do this
np.random.shuffle(training)
train_data = training[:-100]
validation_data = training[-100:]


lr = 1e-2
while lr < 1:
	for ep in range(15,50,5):
		x = tf.placeholder('float')
		y = tf.placeholder('float')
		test_pred, final_accuracy = train_neural_network(x, hm_epochs=ep, learning_rate = lr)
		for i in range(len(test_pred)):
			x = test_pred[i]
			#to avoid overflow
			try:
				x1 = math.exp(x[0])
			except OverflowError:
				x1 = 0
			try:
				x2 = math.exp(x[1])
			except OverflowError:
				x2 = 0
			numerator = max(x1,x2)
			denom = x1 + x2
			if denom == 0:
				p = 1
			else:
				p = numerator/denom
			pred = np.argmax(test_pred[i])
			if pred == 0:
				p = 1 - p
			patient = testing[i][1]
			solution.ix[patient, 'pred'] = p
			solution.ix[patient, 'truth'] = ground_truth.ix[patient,'cancer']
		
		logloss = log_loss(solution.truth, solution.pred)
		print("Epochs: %f, learning rate: %f, accuracy: %f, logloss:  %f" %(ep, lr, final_accuracy, logloss))
		with open(csv_file, 'a') as csvf:
			writer = csv.DictWriter(csvf, fieldnames=field_names, lineterminator='\n')
			writer.writerow({'Epochs number':ep, 'Learning rate':lr, 'accuracy':final_accuracy, 'log loss':logloss})		
	lr *= 10

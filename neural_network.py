import pandas as pd
import numpy as np
from datetime import datetime
import glob
import sys
import re
import os

import utils
from utils import data_importer

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def metadata_importer(path):
	with open(path, 'r') as f:
		df = pd.read_csv(f)

	# change user id from float to str
	df['user_id'] = list(map(lambda x: '0' * (3 - len(str(int(x)))) + str(int(x)), df['user_id']))

	# change datetime from str to datetime.datetime
	df['datetime'] = list(map(lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'), df['end_time'].values))

	print_with_log('Metadata import complete!')
	return df


def label_encoding(y_orig):
	le = preprocessing.LabelEncoder()
	le.fit(y_orig)
	y_orig = le.transform(y_orig)

	return y_orig


def convert_to_one_hot(Y, num_features):
	Y = np.eye(num_features)[Y.reshape(-1)].T
	return Y


def calculate_box_plot_characteristics(my_list: list):
	result = {}
	
	result["minimum"] = np.min(my_list)
	result["maximum"] = np.max(my_list)
	result["median"] = np.median(my_list)
	
	q1 = np.percentile(my_list, 25)
	q3 = np.percentile(my_list, 75)
	iqr = q3 - q1
	result["lower_quartile"] = q1
	result["upper_quartile"] = q3
	
	lower_whisker = q1 - 1.5 * iqr
	upper_whisker = q3 + 1.5 * iqr
	rpa_sort = np.sort(my_list)
	for i in range(len(rpa_sort)):
		if rpa_sort[i] > lower_whisker:
			result["lower_whisker"] = rpa_sort[i]
			break
	for i in reversed(range(len(rpa_sort))):
		if rpa_sort[i] < upper_whisker:
			result["upper_whisker"] = rpa_sort[i]
			break
	
	return result


def initialize_parameters(params_size: list):
	"""
	Initializes parameters to build a neural network with tensorflow.

	Returns:
	parameters -- a dictionary of tensors containing list W and list b
	"""

	W = [None for i in range(len(params_size) - 1)]
	b = [None for i in range(len(params_size) - 1)]

	for i in range(len(params_size) - 1):
		W[i] = tf.get_variable('W'+str(i), [params_size[i+1], params_size[i]], initializer = tf.contrib.layers.xavier_initializer())
		b[i] = tf.get_variable('b'+str(i), [params_size[i+1], 1], initializer = tf.zeros_initializer())

	parameters = {'W': W, 'b': b}
	
	return parameters


def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model: (LINEAR -> RELU) * k -> LINEAR -> SOFTMAX
	
	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing parameters list W and list b

	Returns:
	Z_out -- the output of the last LINEAR unit
	"""
	
	# Retrieve the parameters from the dictionary "parameters" 
	W = parameters['W']
	b = parameters['b']

	Z = [None for i in range(len(W) - 1)]
	A = [None for i in range(len(W) - 1)]

	for i in range(len(W)):
		if i == 0:
			Z[i] = tf.matmul(W[i], X) + b[i]
			A[i] = tf.nn.relu(Z[i])
		elif i != len(W) - 1:
			Z[i] = tf.matmul(W[i], A[i-1]) + b[i]
			A[i] = tf.nn.relu(Z[i])
		else:
			Z_out = tf.matmul(W[i], A[i-1]) + b[i]

	return Z_out


def compute_cost(Z_out, Y, parameters, lambd, m):
	"""
	Compute cost using L2 regularization.
	
	Arguments:
	Z_out -- the output of the last LINEAR unit
	Y -- labels placeholder, of shape (number of classes, number of examples)
	parameters -- python dictionary containing parameters list W and list b
	lambd -- L2 Regularization parameter
	m -- number of examples used to calculate L2 Regularization

	Returns:
	cost -- cost with L2 regularization
	"""
	# Retrieve the parameters from the dictionary "parameters" 
	W = parameters['W']
	
	logits = tf.transpose(Z_out)
	labels = tf.transpose(Y)
	
	cost_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	if lambd != None:
		cost_L2_regularization = 1/m * lambd/2 * np.sum([tf.square(tf.norm(W[i], ord=2)) for i in range(len(W))])
	else:
		cost_L2_regularization = 0
	cost = cost_cross_entropy + cost_L2_regularization
	
	return cost


def random_mini_batches(X, Y, mini_batch_size):
	"""
	Creates a list of random minibatches from (X, Y)
	
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector
	mini_batch_size - size of the mini-batches, integer
	
	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""
	
	m = X.shape[1]                  # number of training examples
	mini_batches = []
	# np.random.seed(seed)
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches


def model(X_train, Y_train, X_test, Y_test, params_size, learning_rate = 0.005,
		  num_epochs = 100, minibatch_size = 1024, lambd = 0.2, print_cost = True, continue_flag=False):
	"""
	Implements a neural network: (LINEAR -> RELU) * k -> LINEAR -> SOFTMAX.
	
	Arguments:
	X_train -- training set
	Y_train -- test set
	X_test -- training set
	Y_test -- test set
	learning_rate -- learning rate of the optimization
	num_epochs -- number of epochs of the optimization loop
	minibatch_size -- size of a minibatch
	lambd -- L2 Regularization parameter
	print_cost -- True to print the cost every epochs
	continue_flag -- True to continue from last checkpoint. If continue, make sure that you have only one model in the folder ./nn_mode
	"""
	
	ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
	(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
	n_y = Y_train.shape[0]                            # n_y : output size
	
	# Create Placeholders of shape (n_x, n_y)
	X = tf.placeholder(tf.float32, [n_x, None], name='X')
	Y = tf.placeholder(tf.float32, [n_y, None], name='Y')

	parameters = initialize_parameters(params_size)
	Z_out = forward_propagation(X, parameters)
	cost = compute_cost(Z_out, Y, parameters, lambd, minibatch_size)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	
	saver = tf.train.Saver(max_to_keep=0)
	
	# find the number of epochs of last checkpoint
	start_epoch = 0
	if continue_flag:
		for file in glob.glob('nn_model/*.index'):
			start_epoch = int(re.findall('[0-9]{3,4}', file)[0])

	# Start the session to compute the tensorflow graph
	with tf.Session() as sess:
		if not continue_flag:
			sess.run(init)
		else:
			saver.restore(sess, tf.train.latest_checkpoint('nn_model'))
		
		# Do the training loop
		for epoch in range(start_epoch, num_epochs + 1):

			epoch_cost = 0.                       # Defines a cost related to an epoch
			num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:
				# Select a minibatch
				(minibatch_X, minibatch_Y) = minibatch
				
				_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches

			# Print the cost every 10 epochs
			if print_cost == True and epoch % 10 == 0:
				print_with_log("Cost after epoch {}: {}".format(epoch, epoch_cost))
				write_cost(epoch_cost)
				
			# Save the model every 100 epochs
			if print_cost == True and epoch % 100 == 0:
				# saver.save(sess, 'nn_model/save_net', global_step=epoch)
				print_with_log('----------------------------------')
				correct_prediction = tf.equal(tf.argmax(Z_out), tf.argmax(Y))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

				print_with_log("Train Accuracy: {}".format(accuracy.eval({X: X_train, Y: Y_train})))
				print_with_log("Test Accuracy: {}".format(accuracy.eval({X: X_test, Y: Y_test})))
				# print_with_log("Saved checkpoint!")
				print_with_log('----------------------------------')

		# Save the parameters in a variable
		parameters = sess.run(parameters)
		print_with_log("Parameters have been trained!")
		

def compute_accuracies(X_train, Y_train, X_test, Y_test, params_size):
	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	
	# Create Placeholders of shape (n_x, n_y)
	X = tf.placeholder(tf.float32, [n_x, None], name='X')
	Y = tf.placeholder(tf.float32, [n_y, None], name='Y')

	parameters = initialize_parameters(params_size)
	Z_out = forward_propagation(X, parameters)
	cost = compute_cost(Z_out, Y, parameters, lambd, minibatch_size)
	saver = tf.train.Saver(max_to_keep=0)

	with tf.Session() as sess:
		saver.restore(sess, tf.train.latest_checkpoint('nn_model'))
		correct_prediction = tf.equal(tf.argmax(Z_out), tf.argmax(Y))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print_with_log("Train Accuracy: {}".format(accuracy.eval({X: X_train, Y: Y_train})))
		print_with_log("Test Accuracy: {}".format(accuracy.eval({X: X_test, Y: Y_test})))


def training(train_index, log_path, csv_path):
	with open(log_path + 'log.txt', 'w') as f:
		f.write('----------- Training #{} ------------\n'.format(train_index))
		# f.write('----------- learning_rate: {} ------------\n'.format(learning_rate))
	with open(log_path + 'costs.txt', 'w') as f:
		f.write('Training #{}\n'.format(train_index))

	df = metadata_importer(csv_path)

	X_orig = df.iloc[:, :4]
	y_orig = df.iloc[:, 4]

	# label encoding
	y_orig = label_encoding(y_orig)
	print_with_log('-----------------------------------')
	print_with_log('shape of X_orig: {}'.format(X_orig.shape))
	print_with_log('shape of y_orig: {}'.format(y_orig.shape))

	# # train set and test set spliting 
	X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.3)
	print_with_log('shape of X_train_orig: {}'.format(X_train_orig.shape))
	print_with_log('shape of y_train_orig: {}'.format(y_train_orig.shape))
	print_with_log('shape of X_test_orig: {}'.format(X_test_orig.shape))
	print_with_log('shape of y_test_orig: {}'.format(y_test_orig.shape))
	print_with_log('-----------------------------------')

	mode_set = set(df['mode'].values)

	# Flatten the training and test images
	X_train = np.array(X_train_orig).reshape(X_train_orig.shape[0], -1).T
	X_test = np.array(X_test_orig).reshape(X_test_orig.shape[0], -1).T

	# Convert training and test labels to one hot matrices
	y_train = convert_to_one_hot(np.array(y_train_orig), len(mode_set))
	y_test = convert_to_one_hot(np.array(y_test_orig), len(mode_set))

	print_with_log("number of training examples = " + str(X_train.shape[1]))
	print_with_log("number of test examples = " + str(X_test.shape[1]))
	print_with_log("X_train shape: " + str(X_train.shape))
	print_with_log("Y_train shape: " + str(y_train.shape))
	print_with_log("X_test shape: " + str(X_test.shape))
	print_with_log("Y_test shape: " + str(y_test.shape))
	print_with_log('-----------------------------------')

	# normalization
	for i in range(len(X_train)):
		X_train[i] = X_train[i] / calculate_box_plot_characteristics(X_train[i])['upper_whisker']
	for i in range(len(X_test)):
		X_test[i] = X_test[i] / calculate_box_plot_characteristics(X_test[i])['upper_whisker']

	# define hyperparameters
	hyperparams = {'params_size': [4, 8, 16, 32, 64, 128, 64, 32, 16, 11], 
			   'learning_rate': 0.001, 
			   'num_epochs': 2000, 
			   'minibatch_size': 128, 
			   'lambda': 0.1}
	print_with_log('hyperparams:')
	for item in hyperparams.keys():
		print_with_log(' - ' + item + ': ' + str(hyperparams[item]))
	print_with_log('-----------------------------------')
	print_with_log('-------------Training--------------')
	print_with_log('-----------------------------------')

	# train the model
	model(X_train, y_train, X_test, y_test, 
			hyperparams['params_size'], hyperparams['learning_rate'], hyperparams['num_epochs'], 
			hyperparams['minibatch_size'], hyperparams['lambda'], continue_flag=False)

	os.rename(log_path + 'log.txt', log_path + 'log #{}.txt'.format(train_index))
	os.rename(log_path + 'costs.txt', log_path + 'costs #{}.txt'.format(train_index))


def print_with_log(output):
	log_path = './nn_model/'
	print(output)
	with open(log_path + 'log.txt', 'a') as f:
		f.write(output + '\n')


def write_cost(epoch_cost):
	log_path = './nn_model/'
	with open(log_path + 'costs.txt', 'a') as f:
		f.write(str(epoch_cost) + '\n')


def main():
	csv_path = './metadata_df.csv'
	log_path = './nn_model/'

	user_input = input("Training #")
	try:
		train_index = int(user_input)
	except:
		raise ValueError("Not an integer!")

	if not os.path.exists(log_path):
		os.makedirs(log_path)

	# tuning learning rate, from 0.0001 to 1, take logrithmic metric, do 20 times tests
	# for train_index in range(1, 21):
	# 	r = -4 * np.random.rand()
	# 	learning_rate = 10 ** r
	# 	training(train_index, log_path, csv_path, learning_rate)

	training(train_index, log_path, csv_path)


if __name__ == "__main__":
	main()

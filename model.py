# -*- coding = utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import numpy as np
import tensorflow as tf

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_CLASSES = read_data.NUM_CLASSES

DROP_PROB = 0.5
REG_STRENGTH = 0.001
INITIAL_LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 5
MOVING_AVERAGE_DECAY = 0.9999
BATCH_SIZE = 100

def _activation_summary(x):
	tf.summary.histogram(x.op.name + '/activations', x)
	tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
	var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='reg_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def inference(images):
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',shape=[11,11,3,96],stddev=1/np.sqrt(11*11*3),wd=0.0)
		weights = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
		biases = tf.get_variable('biases',shape=[96],initializer=tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(weights,biases)
		conv1 = tf.nn.relu(bias,name=scope.name)
		_activation_summary(conv1)
	#pool1
	pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
	#conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',shape=[5,5,96,256],stddev=1/np.sqrt(5*5*96),wd=0.0)
		weights = tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
		biases = tf.get_variable('biases',shape=[256],initializer=tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(weights,biases)
		conv2 = tf.nn.relu(bias,name=scope.name)
		_activation_summary(conv2)
	#pool2
	pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
	#conv3
	with tf.variable_scope('conv3') as scope:
		kernel = _variable_with_weight_decay('weights',shape=[3,3,256,384],stddev=1/np.sqrt(3*3*256),wd=0.0)
		weights = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
		biases = tf.get_variable('biases',shape=[384],initializer=tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(weights,biases)
		conv3 = tf.nn.relu(bias,name=scope.name)
		_activation_summary(conv3)
	#conv4
	with tf.variable_scope('conv4') as scope:
		kernel = _variable_with_weight_decay('weights',shape=[3,3,384,384],stddev=1/np.sqrt(3*3*384),wd=0.0)
		weights = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
		biases = tf.get_variable('biases',shape=[384],initializer=tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(weights,biases)
		conv4 = tf.nn.relu(bias,name=scope.name)
		_activation_summary(conv4)
	#conv5
	with tf.variable_scope('conv5') as scope:
		kernel = _variable_with_weight_decay('weights',shape=[3,3,384,256],stddev=1/np.sqrt(3*3*384),wd=0.0)
		weights = tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
		biases = tf.get_variable('biases',shape=[256],initializer=tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(weights,biases)
		conv5 = tf.nn.relu(bias,name=scope.name)
		_activation_summary(conv5)
	#pool3
	pool3 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool3')
	#full1
	with tf.variable_scope('full1') as scope:
		batch_size = images.get_shape()[0].value
		pool3_flat = tf.reshape(pool3,[batch_size,-1])
		dim = pool3_flat.get_shape()[1].value
		weights = _variable_with_weight_decay('weights',shape=[dim,384],stddev=1/np.sqrt(dim),wd=REG_STRENGTH)
		biases = tf.get_variable('biases',shape=[384],initializer=tf.constant_initializer(0.0))
		full1 = tf.nn.relu(tf.matmul(pool3_flat,weights) + biases,name=scope.name)
		_activation_summary(full1)
	#dropout
	full1_drop = tf.nn.dropout(full1,DROP_PROB)
	#full2
	with tf.variable_scope('full2') as scope:
		weights = _variable_with_weight_decay('weights',shape=[384,192],stddev=1/np.sqrt(384),wd=REG_STRENGTH)
		biases = tf.get_variable('biases',shape=[192],initializer=tf.constant_initializer(0.0))
		full2 = tf.nn.relu(tf.matmul(full1_drop,weights)+biases,name=scope.name)
		_activation_summary(full2)
	#dropout
	full2_drop = tf.nn.dropout(full2,DROP_PROB)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights',shape=[192,NUM_CLASSES],stddev=1/np.sqrt(192),wd=REG_STRENGTH)
		biases = tf.get_variable('biases',shape=[NUM_CLASSES],initializer=tf.constant_initializer(0.0))
		logits = tf.add(tf.matmul(full2_drop,weights),biases,name=scope.name)
		_activation_summary(logits)

	return logits

def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits, name='xentropy')
	data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	tf.add_to_collection('losses', data_loss)
	total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
	return total_loss

def _loss_summaries(total_loss):
	losses = tf.get_collection('losses')
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name, l)

def training(total_loss):

	global_step = tf.Variable(0, name='global_step', trainable=False)
	decay_steps = int(EPOCHS_PER_LR_DECAY * NUM_TRAIN_EXAMPLES / BATCH_SIZE)
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LR_DECAY_FACTOR, staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)

	_loss_summaries(total_loss)

	optimizer = tf.train.AdamOptimizer(learning_rate)
	opt_op = optimizer.minimize(total_loss, global_step=global_step)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	mov_average_object = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	moving_average_op = mov_average_object.apply(tf.trainable_variables())

	with tf.control_dependencies([opt_op]):
		train_op = tf.group(moving_average_op)

	return train_op


def evaluation(logits, true_labels):
	correct_pred = tf.nn.in_top_k(logits, true_labels, 1)
	return tf.reduce_mean(tf.cast(correct_pred, tf.float32))*100

from __future__ import print_function
import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data/', one_hot = True)
import config as cfg 

class LSTM(object):
	def __init__(self):
		self.learning_rate  = cfg.LEARNING_RATE 
		self.training_steps = cfg.TRAINING_STEPS 
		self.batch_size     = cfg.BATCH_SIZE 
		self.display_step   = cfg.DISPLAY_STEP 
		self.num_input      = cfg.num_input
		self.timesteps      = cfg.TIMESTEPS 
		self.num_hidden     = cfg.NUM_HIDDEN 
		self.num_classes    = cfg.NUM_CLASSES 

	def build_model(self):
		with tf.variable_scope('RNN') as scope:
			self.inputs = tf.placeholder(shape = [None, self.timesteps, self.num_input], dtype = tf.float32, name = 'inputs')
			self.labels = tf.placeholder(shape = [None, self.num_classes], dtype = tf.float32, name = 'labels')
			weights = tf.get_variable(shape = [self.num_hidden, self.num_classes], dtype = tf.float32, initializer = tf.random_normal_initializer(), name = 'weights')
			bias    = tf.get_variable(shape = [self.num_classes])
			x = tf.unstack(inputs, self.timesteps, axis = 1)
			lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias = 1.0)
			outputs, state = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
			logits = 
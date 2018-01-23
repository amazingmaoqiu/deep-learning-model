from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data/', one_hot = True)
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import config as cfg

class CNN(object):
	def __init__(self):
		self.learning_rate = cfg.LEARNING_RATE 
		self.num_steps     = cfg.NUM_STEPS 
		self.batch_size    = cfg.BATCH_SIZE 
		self.num_input     = cfg.NUM_INPUT 
		self.num_classes   = cfg.NUM_CLASSES 
		self.display_step  = cfg.DISPLAY_STEP
		self.dropout       = cfg.DROPOUT 


	def build_model(self, reuse, is_training):
		print('start to build model.')
		with tf.variable_scope('ConvNet', reuse = reuse) as scope:
			self.inputs = tf.placeholder(shape = [None, 784], dtype = tf.float32, name = 'inputs')
			self.labels = tf.placeholder(shape = [None, 10],  dtype = tf.float32, name = 'labels')
			x = tf.reshape(self.inputs, shape = [-1, 28, 28, 1])
			conv1 = tf.layers.conv2d(x, 32, 5, activation = tf.nn.relu)
			conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

			conv2 = tf.layers.conv2d(conv1, 64, 3, activation = tf.nn.relu)
			conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

			fc1 = tf.contrib.layers.flatten(conv2)
			fc1 = tf.layers.dense(fc1, 1024)
			fc1 = tf.layers.dropout(fc1, rate = self.dropout, training = is_training)

			out = tf.layers.dense(fc1, self.num_classes)

		with tf.variable_scope('loss_and_accuracy') as scope:
			self.sess  = tf.Session()
			self.loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = tf.cast(self.labels, dtype = tf.float32)))
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
			self.init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.writer = tf.summary.FileWriter('my_graph', self.sess.graph)
			self.writer.close()
		print('model completed.')
		return out

	def train(self):
		predict = self.build_model(reuse = False, is_training = True)
		self.sess.run(self.init)
		for epoch in range(10):
			avg_loss = 0
			total_batch = int(mnist.train.num_examples // self.batch_size)
			for batch in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(self.batch_size)
				_, cost = self.sess.run((self.optimizer, self.loss), feed_dict = {self.inputs:batch_x, self.labels:batch_y})
				avg_loss += cost / total_batch
			if(epoch % self.display_step == 0):
				print("epoch %04d : loss = %.9f"%(epoch, avg_loss))
		self.saver.save(self.sess, 'model/cnn.ckpt', global_step = 5)
		print("training completed.")

	def test(self):
		predict = self.build_model(reuse = False, is_training = False)
		self.saver.restore(self.sess, 'model/cnn.ckpt-5')
		correct_prediction = tf.equal(tf.argmax(predict, axis = 1), tf.argmax(self.labels, axis = 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
		total_batch = int(mnist.test.num_examples // self.batch_size)
		acc = 0
		for batch in range(total_batch):
			batch_x, batch_y = mnist.test.next_batch(self.batch_size)
			acc_batch, prediction = self.sess.run((accuracy, correct_prediction), feed_dict = {self.inputs:batch_x, self.labels:batch_y})
			acc += acc_batch / total_batch
		print("accuracy = %.9f"%acc)
		print("test completed.")

	def retrain(self):
		predict = self.build_model(reuse = False, is_training = True)
		self.saver.restore(self.sess, 'model/cnn.ckpt-5')
		for epoch in range(10):
			avg_loss = 0
			total_batch = int(mnist.train.num_examples // self.batch_size)
			for batch in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(self.batch_size)
				_, cost = self.sess.run((self.optimizer, self.loss), feed_dict = {self.inputs:batch_x, self.labels:batch_y})
				avg_loss += cost / total_batch
			if(epoch % self.display_step == 0):
				print("epoch %04d : loss = %.9f"%(epoch, avg_loss))
		self.saver.save(self.sess, 'model/cnn.ckpt', global_step = 5)
		print("training completed.")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	args  = parser.parse_args()
	model = CNN()
	if(args.mode == 'train'):
		model.train()
	if(args.mode == 'retrain'):
		model.retrain()
	if(args.mode == 'test'):
		model.test()
	else:
		raise Exception('mode wrong. you should input mode as train or retain or test.')

if __name__ == "__main__":
	main()




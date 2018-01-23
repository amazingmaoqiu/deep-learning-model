from __future__ import division, print_function, absolute_import 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import tensorflow as tf 
import config as cfg 
import argparse
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('../MNIST_data/', one_hot = True)


class AUTO_ENCODER(object):
	def __init__(self):
		self.learning_rate = cfg.LEARNING_RATE 
		self.num_steps     = cfg.NUM_STEPS 
		self.batch_size    = cfg.BATCH_SIZE 
		self.display_step  = cfg.DISPLAY_STEP 
		self.examples_to_show = cfg.EXAMPLES_TO_SHOW 
		self.num_hidden_1  = cfg.NUM_HIDDEN_1 
		self.num_hidden_2  = cfg.NUM_HIDDEN_2 
		self.num_input     = cfg.NUM_INPUT 
		self.X             = tf.placeholder(shape = [None, self.num_input], dtype = tf.float32)
		self.weights       = {'encoder_h1':tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1]), trainable = False),
		                      'encoder_h2':tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2]), trainable = False),
		                      'decoder_h1':tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
		                      'decoder_h2':tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input]))
		                      }
		self.biases        = {'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
							  'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
							  'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
							  'decoder_b2': tf.Variable(tf.random_normal([self.num_input]))
							  }

	def encoder(self, x):
		with tf.variable_scope('ENCODER', reuse = False) as scope:
			layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
			layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
			return layer_2

	def decoder(self, x):
		with tf.variable_scope('DECODER', reuse = False):
			layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
			layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
			return layer_2

	def build_model(self):
		self.encoder_op = self.encoder(self.X)
		self.decoder_op = self.decoder(self.encoder_op)
		y_pred     = self.decoder_op
		y_true     = self.X
		self.sess = tf.Session()
		self.loss  = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter('my_graph', self.sess.graph)
		self.writer.close()
		print('model completed.')

	def train(self):
		self.sess.run(self.init)
		for i in range(1, self.num_steps+1):
			batch_x, _ = mnist.train.next_batch(self.batch_size)
			_, l = self.sess.run([self.optimizer, self.loss], feed_dict = {self.X:batch_x})
			if i%self.display_step == 0 or i == 1:
				print('Step %i: Minibatch Loss: %f' %(i, l))
		self.saver.save(self.sess, 'model/auto_encoder.ckpt', global_step = 5)

	def test(self):
		model = self.build_model()
		self.saver.restore(self.sess, 'model/auto_encoder.ckpt-5')
		n = 4
		canvas_orig = np.empty((28 * n, 28 * n))
		canvas_recon = np.empty((28 * n, 28 * n))
		for i in range(n):
			batch_x, _ = mnist.train.next_batch(n)
			g = self.sess.run(self.decoder_op, feed_dict = {self.X:batch_x})
			for j in range(n):
				canvas_orig[i*28:(i+1)*28, j*28:(j+1)*28] = batch_x[j].reshape([28,28])
			for j in range(n):
				canvas_recon[i*28:(i+1)*28, j*28:(j+1)*28] = g[j].reshape([28, 28])
		print('original images')
		plt.figure(figsize = (n, n))
		plt.imshow(canvas_orig, origin = "upper", cmap = "gray")
		plt.show()

		print('Reconstructed images')
		plt.figure(figsize = (n, n))
		plt.imshow(canvas_recon, origin = "upper", cmap = "gray")
		plt.show()

	def retrain(self):
		model = self.build_model()
		self.saver.restore(self.sess, 'model/auto_encoder.ckpt-5')
		for i in range(1, self.num_steps+1):
			batch_x, _ = mnist.train.next_batch(self.batch_size)
			_, l = self.sess.run([self.optimizer, self.loss], feed_dict = {self.X:batch_x})
			if i%self.display_step == 0 or i == 1:
				print('Step %i: Minibatch Loss: %f' %(i, l))
		self.saver.save(self.sess, 'model/auto_encoder.ckpt', global_step = 5)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	args = parser.parse_args()
	model = AUTO_ENCODER()
	if(args.mode == 'train'):
		model.train()
	elif(args.mode == 'retrain'):
		model.retrain()
	elif(args.mode == 'test'):
		model.test()
	else:
		raise Exception('mode wrong. you should input mode as train or retrain or test.')

if __name__ == '__main__':
	main()


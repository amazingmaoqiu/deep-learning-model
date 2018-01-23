from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("../MNIST_data/", one_hot = True)
import config as cfg 

class DCGAN(object):
	def __init__(self):
		self.num_steps        = cfg.NUM_STEPS 
		self.batch_size       = cfg.BATCH_SIZE 
		self.lr_generator     = cfg.LR_GENERATOR 
		self.lr_discriminator = cfg.LR_DISCRIMINATOR 
		self.image_dim        = cfg.IMAGE_DIM 
		self.noise_dim        = cfg.NOISE_DIM 
		self.noise_inputs     = tf.placeholder(shape = [None, 100], dtype = tf.float32, name = 'inputs')
		self.real_image_inputs= tf.placeholder(shape = [None, 28, 28, 1], dtype = tf.float32, name = 'real_image_inputs') 

	def leaky_relu(self, x, alpha):
		return tf.maximum(alpha*x, x)

	def generator(self, is_training, reuse = False):
		with tf.variable_scope('Generator', reuse = reuse) as scope:
			net = tf.layers.dense(self.noise_inputs, units = 7*7*128)
			net = tf.layers.batch_normalization(net, training = is_training)
			net = tf.nn.relu(net)
			net = tf.reshape(net, [-1, 7, 7, 128])
			net = tf.layers.conv2d_transpose(inputs = net, filters = 64, kernel_size = 5, strides = 2, padding = 'SAME')
			net = tf.layers.batch_normalization(net, training = is_training)
			net = tf.nn.relu(net)
			net = tf.layers.conv2d_transpose(inputs = net, filters = 1, kernel_size = 5, strides = 2, padding = 'SAME')
			self.output_generator = tf.nn.tanh(net)
			return self.output_generator

	def discriminator(self, inputs, is_training, reuse = False):
		with tf.variable_scope('Discriminator', reuse = reuse) as scope:
			net = tf.layers.conv2d(inputs = inputs, filters = 64, kernel_size = 5, strides = 2, padding = 'SAME')
			net = tf.layers.batch_normalization(net, training = is_training)
			net = self.leaky_relu(net, 0.2)
			net = tf.layers.conv2d(net, filters = 128, kernel_size = 5, strides = 2, padding = 'SAME')
			net = tf.layers.batch_normalization(net, training = is_training)
			net = self.leaky_relu(net, 0.2)
			net = tf.reshape(net, [-1, 7*7*128])
			net = tf.layers.dense(net, 1024)
			net = tf.layers.batch_normalization(net, training = is_training)
			net = self.leaky_relu(net, 0.2)
			output_dsicriminator = tf.layers.dense(net, 2)
			return output_dsicriminator

	def build_model(self):
		gen_sample = self.generator(is_training = True)
		disc_real  = self.discriminator(self.real_image_inputs, is_training = True, reuse = False)
		disc_fake  = self.discriminator(gen_sample, is_training = True, reuse = True)
		stacked_gan= self.discriminator(gen_sample, is_training = True, reuse = True)

		disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_real, labels = tf.ones([self.batch_size], dtype = tf.int32)))
		disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_fake, labels = tf.zeros([self.batch_size], dtype = tf.int32)))

		self.disc_loss = disc_loss_real + disc_loss_fake
		self.gen_loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan, labels = tf.ones([self.batch_size], dtype = tf.int32)))

		optimizer_gen = tf.train.AdamOptimizer(learning_rate = self.lr_generator, beta1 = 0.5, beta2 = 0.999)
		optimizer_disc = tf.train.AdamOptimizer(learning_rate = self.lr_discriminator, beta1 = 0.5, beta2 = 0.999)

		gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Generator")
		disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

		gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Generator')

		with tf.control_dependencies(gen_update_ops):
			self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list = gen_vars)
		disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Discriminator')
		with tf.control_dependencies(disc_update_ops):
			self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list = disc_vars)
		self.init = tf.global_variables_initializer()
		self.sess = tf.Session()

	def train(self):
		self.sess.run(self.init)
		for i in range(1, self.num_steps+1):
			batch_x, _ = mnist.train.next_batch(self.batch_size)
			batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
			batch_x = batch_x * 2.0 - 1.0
			z = np.random.uniform(-1.0, 1.0, size = [self.batch_size, self.noise_dim])
			_, dl = self.sess.run((self.train_disc, self.disc_loss), feed_dict = {self.real_image_inputs:batch_x, self.noise_inputs:z})
			z = np.random.uniform(-1.0, 1.0, [self.batch_size, self.noise_dim])
			_, gl = self.sess.run((self.train_gen, self.gen_loss), feed_dict = {self.noise_inputs:z})

			if i % 500 == 0 or i == 1:
				print('step %i: Generator Loss: %f, Discriminator Loss: %f'%(i, gl, dl))

def main():
	model = DCGAN()
	model.build_model()
	model.train()

if __name__ == "__main__":
	main()

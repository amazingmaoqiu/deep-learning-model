import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

class vgg16(object):
    def __init__(self, weights = None, sess = None):
        #self.imgs = imgs
        self.sess = tf.Session()
        self.model()
        
        #self.probs = tf.nn.softmax(self.fc3l)
        #if weightsid is not None and sess is not None:
            #self.load_weights(weights, sess)
            
    def conv(self, input, kernel_size, num_filter, layer):
        with tf.name_scope('conv' + str(layer)) as scope:
            channel = int(np.prod(input.get_shape()[-1]))
            kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channel, num_filter], dtype = tf.float32, stddev = 1e-1), name = 'weights')
            conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding = 'SAME')
            biases = tf.Variable(tf.constant(0.0, shape = [num_filter], dtype = tf.float32), trainable = True, name = 'biases')
            out = tf.nn.relu(tf.add(conv, biases))
            self.parameters += [kernel, biases]
        print("conv" + str(layer) + " completed")
        return out
    
    def max_pooling(self, input, kernel_size, stride, layer):
        with tf.name_scope('pool' + str(layer)) as scope:
            out = tf.nn.max_pool(input, ksize = [1, kernel_size, kernel_size, 1], strides = [1, stride, stride, 1], padding = 'SAME', name = 'pool')
        print("pool" + str(layer) + "completed")
        return out
    
    def fc_layer(self, input, layer, out_size):
        with tf.name_scope('fc' + str(layer)) as scope:
            shape = int(np.prod(input.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, out_size], dtype = tf.float32, stddev = 1e-1), name = 'weights')
            biases = tf.Variable(tf.constant(1.0, shape = [out_size], dtype = tf.float32), trainable = True, name = 'biases')
            out = tf.reshape(input, [-1, shape])
            out = tf.add(tf.matmul(out, weights), biases)
            self.parameters += [weights, biases]
        print("fc" + str(layer) + " completed")
        return out
    
    def model(self):
    	escape_list = ['fc8_W', 'fc8_b']
        self.parameters = []
        self.inputs = tf.placeholder(shape = [None, 224, 224, 3], dtype = tf.float32, name = 'inputs')
        self.net = self.conv(self.inputs, 3, 64, 1)
        self.net = self.conv(self.net, 3, 64, 2)
        self.net = self.max_pooling(self.net, 2, 2, 1)
        self.net = self.conv(self.net, 3, 128, 3)
        self.net = self.conv(self.net, 3, 128, 4)
        self.net = self.max_pooling(self.net, 2, 2, 2)
        self.net = self.conv(self.net, 3, 256, 5)
        self.net = self.conv(self.net, 3, 256, 6)
        self.net = self.conv(self.net, 3, 256, 7)
        self.net = self.max_pooling(self.net, 2, 2, 3)
        self.net = self.conv(self.net, 3, 512, 8)
        self.net = self.conv(self.net, 3, 512, 9)
        self.net = self.conv(self.net, 3, 512, 10)
        self.net = self.max_pooling(self.net, 2, 2, 4)
        self.net = self.conv(self.net, 3, 512, 11)
        self.net = self.conv(self.net, 3, 512, 12)
        self.net = self.conv(self.net, 3, 512, 13)
        self.net = self.max_pooling(self.net, 2, 2, 5)
        self.net = self.fc_layer(self.net, 1, 4096)
        self.net = self.fc_layer(self.net, 2, 4096)
        self.net = self.fc_layer(self.net, 3, 1000)
        weights = np.load("vgg16_weights.npz")
        keys = sorted(weights.keys())
        #self.sess.run(self.parameters[0].assign(weights['conv1_1_W']))
        #self.sess.run(self.parameters[1].assign(weights['conv1_1_b']))
        #self.sess.run(self.parameters[2].assign(weights['conv1_2_W']))
        for i, k in enumerate(keys):
            if k not in escape_list:
                self.sess.run(self.parameters[i].assign(weights[k]))
                print(k + " assign completed")
        print("model completed.")
  
def main():
    vgg = vgg16()

    
if __name__ == '__main__':
    main()
    

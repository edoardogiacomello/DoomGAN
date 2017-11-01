import tensorflow as tf
import math
from DoomLevelsGAN.NNHelpers import *

class DoomGAN(object):

    def generator(self, z):
        def calc_filter_size(output_size, stride):
            return [int(math.ceil(float(z) / float(t))) for z, t in zip(output_size,stride)]

        def g_activ_batch_nrm(x, name='g_a'):
            '''Activation function used in the generator, also includes a batch normalization layer '''
            batch_norm_layer = batch_norm(name=name)
            return tf.nn.relu(batch_norm_layer(x))

        with tf.variable_scope("G") as scope:
            # Calculating filter and layer sizes based on expected output
            g_size_filter_h4 = self.output_size
            g_size_filter_h3 = calc_filter_size(output_size=g_size_filter_h4, stride=(2,2))
            g_size_filter_h2 = calc_filter_size(output_size=g_size_filter_h3, stride=(2,2))
            g_size_filter_h1 = calc_filter_size(output_size=g_size_filter_h2, stride=(2,2))
            g_size_filter_h0 = calc_filter_size(output_size=g_size_filter_h1, stride=(2,2))

            g_size_h0 = [-1,              g_size_filter_h0[0], g_size_filter_h0[1], self.g_filter_depth*8]
            g_size_h1 = [self.batch_size, g_size_filter_h1[0], g_size_filter_h1[1], self.g_filter_depth*4]
            g_size_h2 = [self.batch_size, g_size_filter_h2[0], g_size_filter_h2[1], self.g_filter_depth*2]
            g_size_h3 = [self.batch_size, g_size_filter_h3[0], g_size_filter_h3[1], self.g_filter_depth*1]
            g_size_h4 = [self.batch_size, g_size_filter_h4[0], g_size_filter_h4[1], self.output_channels]

            g_size_z_p = g_size_h0[1]*g_size_h0[2]*g_size_h0[3]

            # Projection of Z
            z_p = linear_layer(z, g_size_z_p, 'g_h0_lin')

            g_h0 = g_activ_batch_nrm(tf.reshape(z_p,  g_size_h1), name='g_a0')
            g_h1 = g_activ_batch_nrm(conv2d_transposed(g_h0, g_size_h1, name='g_h1'), name='g_a1')
            g_h2 = g_activ_batch_nrm(conv2d_transposed(g_h1, g_size_h2, name='g_h2'), name='g_a2')
            g_h3 = g_activ_batch_nrm(conv2d_transposed(g_h2, g_size_h3, name='g_h3'), name='g_a3')
            g_h4 = g_activ_batch_nrm(conv2d_transposed(g_h3, g_size_h4, name='g_h4'), name='g_a4')
        return tf.nn.tanh(g_h4)

    def discriminator(self, input, reuse=False):
        def d_activ_batch_norm(x, name="d_a"):
            batch_norm_layer = batch_norm(name)
            return leaky_relu(batch_norm_layer(x))

        with tf.variable_scope("D") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = leaky_relu(conv2d(input, self.d_filter_depth, name='d_a0'))
            h1 = d_activ_batch_norm(conv2d(h0, self.d_filter_depth*2, name='d_a1'))
            h2 = d_activ_batch_norm(conv2d(h1, self.d_filter_depth*4, name='d_a2'))
            h3 = d_activ_batch_norm(conv2d(h2, self.d_filter_depth*8, name='d_a3'))
            h4 = linear_layer(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_a4')
            return tf.nn.sigmoid(h4), h4

    def loss_function(self):
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.loss_d_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_real, tf.ones_like(self.D_real)))
        self.loss_d_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.zeros_like(self.D_fake)))
        self.loss_g = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.ones_like(self.D_fake)))
        self.loss_d = self.loss_d_real + self.loss_d_fake
        return self.loss_d, self.loss_g

    def build(self):
        # x: True inputs coming from the dataset
        # z: Noise in input to the generator

        self.x = tf.placeholder(tf.float32, [self.batch_size] + self.output_size + [self.output_channels], name="real_inputs")
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # Generator network
        self.G = self.generator(self.z)
        # Discriminator networks for each input type (real and generated)
        self.D_real, self.D_logits_real = self.discriminator(self.x, reuse=False)
        self.D_fake, self.D_logits_fake = self.discriminator(self.G, reuse=True)
        # Define the loss function
        self.loss_d, self.loss_g = self.loss_function()
        # Collect the trainable variables for the optimizer
        # TODO: Continue here

    def train(self, config):
        def optimizer(config):
            # TODO: Continue here, need to collect variables for each net
            d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss_d, var_list=self.d_vars)
            g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss_g, var_list=self.g_vars)
            return d_optim, g_optim

    def __init__(self, output_size, output_channels=3, batch_size=64, g_filter_depth=64, d_filter_depth=64, z_dim=100):
        assert len(output_size) == 2, "Data size must have 2 dimensions. Depth is specified in 'output_channels' parameter"
        self.output_size = output_size
        self.output_channels = output_channels
        self.g_filter_depth = g_filter_depth
        self.d_filter_depth = d_filter_depth
        self.z_dim = z_dim
        self.batch_size=batch_size
        self.build()
        pass
    def train(self):
        pass

DoomGAN(output_size=[512,512])
import tensorflow as tf
import math
from DoomLevelsGAN.NNHelpers import *

class DoomGAN(object):

    def generator(self, z):
        def calc_filter_size(output_size, stride):
            return tuple(int(math.ceil(float(z) / float(t))) for z, t in zip(output_size,stride))

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

    def discriminator(self):
        pass

    def build(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        pass

    def __init__(self, output_size, output_channels=3, batch_size=64, g_filter_depth=64, z_dim=100):
        self.output_size = output_size
        self.output_channels = output_channels
        self.g_filter_depth = g_filter_depth
        self.z_dim = z_dim
        self.batch_size=batch_size
        self.build()
        pass
    def train(self):
        pass

DoomGAN(output_size=(512,512))
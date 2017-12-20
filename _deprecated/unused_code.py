def generator(self, z):
    '''
    Generator for the DCGAN architecture (unaltered - rewritten for readability)
    :param z: noise input to the net
    :return:
    '''

    def calc_filter_size(output_size, stride):
        return [int(math.ceil(float(z) / float(t))) for z, t in zip(output_size, stride)]

    def g_activ_batch_nrm(x, name='g_a'):
        '''Activation function used in the generator, also includes a batch normalization layer '''
        batch_norm_layer = batch_norm(name=name)
        return tf.nn.relu(batch_norm_layer(x))

    with tf.variable_scope("G") as scope:
        # Calculating filter and layer sizes based on expected output
        g_size_filter_h4 = self.output_size
        g_size_filter_h3 = calc_filter_size(output_size=g_size_filter_h4, stride=(2, 2))
        g_size_filter_h2 = calc_filter_size(output_size=g_size_filter_h3, stride=(2, 2))
        g_size_filter_h1 = calc_filter_size(output_size=g_size_filter_h2, stride=(2, 2))
        g_size_filter_h0 = calc_filter_size(output_size=g_size_filter_h1, stride=(2, 2))

        g_size_h0 = [-1, g_size_filter_h0[0], g_size_filter_h0[1], self.g_filter_depth * 8]
        g_size_h1 = [self.batch_size, g_size_filter_h1[0], g_size_filter_h1[1], self.g_filter_depth * 4]
        g_size_h2 = [self.batch_size, g_size_filter_h2[0], g_size_filter_h2[1], self.g_filter_depth * 2]
        g_size_h3 = [self.batch_size, g_size_filter_h3[0], g_size_filter_h3[1], self.g_filter_depth * 1]
        g_size_h4 = [self.batch_size, g_size_filter_h4[0], g_size_filter_h4[1], self.output_channels]

        g_size_z_p = g_size_h0[1] * g_size_h0[2] * g_size_h0[3]

        # Projection of Z
        z_p = linear_layer(z, g_size_z_p, 'g_h0_lin')

        g_h0 = g_activ_batch_nrm(tf.reshape(z_p, g_size_h0))
        g_h1 = g_activ_batch_nrm(conv2d_transposed(g_h0, g_size_h1, name='g_h1'), name='g_a1')
        g_h2 = g_activ_batch_nrm(conv2d_transposed(g_h1, g_size_h2, name='g_h2'), name='g_a2')
        g_h3 = g_activ_batch_nrm(conv2d_transposed(g_h2, g_size_h3, name='g_h3'), name='g_a3')
        g_h4 = conv2d_transposed(g_h3, g_size_h4, name='g_h4')
    return tf.nn.tanh(g_h4)


def discriminator(self, input, reuse=False):
    def d_activ_batch_norm(x, name="d_a"):
        batch_norm_layer = batch_norm(name=name)
        return leaky_relu(batch_norm_layer(x))

    with tf.variable_scope("D") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = leaky_relu(conv2d(input, self.d_filter_depth, name='d_a0'))
        h1 = d_activ_batch_norm(conv2d(h0, self.d_filter_depth * 2, name='d_h1', k_w=3, k_h=3), name='d_a1')
        h2 = d_activ_batch_norm(conv2d(h1, self.d_filter_depth * 4, name='d_h2', k_w=3, k_h=3), name='d_a2')
        h3 = d_activ_batch_norm(conv2d(h2, self.d_filter_depth * 8, name='d_h3', k_w=3, k_h=3), name='d_a3')
        h4 = linear_layer(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_a4')
        return tf.nn.sigmoid(h4), h4
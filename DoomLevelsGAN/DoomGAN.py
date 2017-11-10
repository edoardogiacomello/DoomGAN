import tensorflow as tf
import os
import numpy as np
import math
from DoomLevelsGAN.NNHelpers import *
from dataset_utils import DatasetManager
class DoomGAN(object):
    def generator_generalized(self, z, hidden_layers):
        '''
        Parametrized version of the DCGAN generator, accept a variable number of hidden layers
        :param z: Noise as input to the net
        :param hidden_layers: The number of hidden layer to use. E.g. "5" will produce layers from h0 to h5 (output)
        :param stride: Convolutional stride to use for all the layers (tuple), default: (2,2)
        :return:
        '''
        def calc_filter_size(output_size, stride):
            return [int(math.ceil(float(z) / float(t))) for z, t in zip(output_size,stride)]

        def g_activ_batch_nrm(x, name='g_a'):
            '''Activation function used in the generator, also includes a batch normalization layer '''
            batch_norm_layer = batch_norm(name=name)
            return tf.nn.relu(batch_norm_layer(x))


        with tf.variable_scope("G") as scope:
            # Calculating filter size
            g_size_filter = []
            for reverse_id, layer in enumerate(reversed(hidden_layers)):
                if reverse_id == 0:  # here the id is reversed, so last layer has index 0
                    g_size_filter.insert(0, self.output_size)
                else:
                    g_size_filter.insert(0, calc_filter_size(output_size=g_size_filter[0], stride=layer['stride']))
            # Calculating layer size
            g_size = []
            for layer_id, layer in enumerate(hidden_layers):
                size = [self.batch_size, g_size_filter[layer_id][0], g_size_filter[layer_id][1], layer['n_filters']]
                # First and last layer differs from the others
                if layer_id == 0:
                    size[0] = -1
                if layer_id == len(hidden_layers) - 1:
                    size[-1] = self.output_channels
                g_size.append(size)
            # Size for the Z projection
            g_size_z_p = g_size[0][1] * g_size[0][2] * g_size[0][3]
            # Projection of Z
            z_p = linear_layer(z, g_size_z_p, 'g_h0_lin')
            self.layers_G = []
            for layer_id, layer in enumerate(hidden_layers):
                if layer_id == 0:
                    l = g_activ_batch_nrm(tf.reshape(z_p, g_size[0]))
                else:
                    if layer_id == len(hidden_layers) - 1:
                        l = conv2d_transposed(self.layers_G[layer_id-1], g_size[layer_id], name='g_h{}'.format(layer_id),
                                              stride_h=layer['stride'][0], stride_w=layer['stride'][1],
                                              k_h = layer['kernel_size'][0], k_w = layer['kernel_size'][1]
                        )
                    else:
                        l = g_activ_batch_nrm(conv2d_transposed(self.layers_G[layer_id-1], g_size[layer_id], name='g_h{}'.format(layer_id),
                                                                stride_h=layer['stride'][0], stride_w=layer['stride'][1],
                                                                k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                                                ), name='g_a{}'.format(layer_id))
                self.layers_G.append(l)
        return tf.nn.tanh(self.layers_G[-1])

    def discriminator_generalized(self, input, hidden_layers, reuse=False):
        def d_activ_batch_norm(x, name="d_a"):
            batch_norm_layer = batch_norm(name=name)
            return leaky_relu(batch_norm_layer(x))
        with tf.variable_scope("D") as scope:
            if reuse:
                scope.reuse_variables()
            self.layers_D = []
            for layer_id, layer in enumerate(hidden_layers):
                if layer_id == 0:  # First layer (input)
                    l = leaky_relu(conv2d(input, layer['n_filters'], name='d_a{}'.format(layer_id),
                                          k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                          stride_h=layer['stride'][0], stride_w=layer['stride'][1]))
                else:
                    if layer_id == len(hidden_layers)-1:  # Last layer (output)
                        l = linear_layer(tf.reshape(self.layers_D[-1], [self.batch_size, -1]), 1, 'd_a{}'.format(layer_id))
                    else:  # Hidden layers
                        l = d_activ_batch_norm(conv2d(self.layers_D[-1], layer['n_filters'], name="d_h{}".format(layer_id),
                                                      k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                                      stride_h=layer['stride'][0], stride_w=layer['stride'][1]),
                                               name="g_a{}".format(layer_id))
                self.layers_D.append(l)
        return tf.nn.sigmoid(self.layers_D[-1]), self.layers_D[-1]


    def generator(self, z):
        '''
        Generator for the DCGAN architecture (unaltered - rewritten for readability)
        :param z: noise input to the net
        :return:
        '''
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

            g_h0 = g_activ_batch_nrm(tf.reshape(z_p,  g_size_h0))
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
            h1 = d_activ_batch_norm(conv2d(h0, self.d_filter_depth*2, name='d_h1', k_w=3, k_h=3), name='d_a1')
            h2 = d_activ_batch_norm(conv2d(h1, self.d_filter_depth*4, name='d_h2', k_w=3, k_h=3), name='d_a2')
            h3 = d_activ_batch_norm(conv2d(h2, self.d_filter_depth*8, name='d_h3', k_w=3, k_h=3), name='d_a3')
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
        self.x_norm = (self.x-tf.constant(127.5, dtype=tf.float32))/tf.constant(127.5, dtype=tf.float32) if self.normalize_input else self.x
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        # Generator network
        g_layers = [
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128},
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128},
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128},
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128},
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128},
            {'stride': (2,2), 'kernel_size': (3,3), 'n_filters': 128}
        ]

        d_layers = [
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64},
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64},
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64},
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64},
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64},
            {'stride': (2, 2), 'kernel_size': (3, 3), 'n_filters': 64}
        ]

        self.G = self.generator_generalized(self.z, hidden_layers=g_layers)
        #self.G = self.generator(self.z)
        # Discriminator networks for each input type (real and generated)
        self.D_real, self.D_logits_real = self.discriminator_generalized(self.x_norm, d_layers, reuse=False)
        self.D_fake, self.D_logits_fake = self.discriminator_generalized(self.G, d_layers, reuse=True)
        # Define the loss function
        self.loss_d, self.loss_g = self.loss_function()
        # Collect the trainable variables for the optimizer
        vars = tf.trainable_variables()
        self.vars_d = [var for var in vars if 'd_' in var.name]
        self.vars_g = [var for var in vars if 'g_' in var.name]

    def generate_summary(self):
        s_loss_d_real = tf.summary.scalar('d_loss_real_inputs', self.loss_d_real)
        s_loss_d_fake = tf.summary.scalar('d_loss_fake_inputs', self.loss_d_fake)
        s_loss_d = tf.summary.scalar('d_loss', self.loss_d)
        s_loss_g = tf.summary.scalar('g_loss', self.loss_g)
        s_z_distrib = tf.summary.histogram('z_distribution', self.z)
        s_sample = visualize_samples('generated_samples', self.G)
        s_input = visualize_samples('true_samples', self.x_norm)


        s_d = tf.summary.merge([s_loss_d_real, s_loss_d_fake, s_loss_d])
        s_g = tf.summary.merge([s_loss_g, s_z_distrib])
        s_samples = tf.summary.merge([s_sample, s_input])

        summary_writer = tf.summary.FileWriter(self.summary_folder)

        return s_d, s_g, s_samples, summary_writer

    def save(self, checkpoint_dir):
        # Code from https://github.com/carpedm20/DCGAN-tensorflow
        model_name = "DOOMGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=self.checkpoint_counter)

    def load(self, checkpoint_dir):
        # Code from https://github.com/carpedm20/DCGAN-tensorflow
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_dataset(self):
        """
        Loads a TFRecord dataset and creates batches.
        :return: An initializable Iterator for the loaded dataset
        """
        self.dataset = DatasetManager(target_size=self.output_size).load_TFRecords_database(self.dataset_path)
        # If the dataset size is unknown, it must be retrieved (tfrecords doesn't hold metadata and the size is needed
        # for discarding the last incomplete batch)
        if self.dataset_size is None:
            counter_iter = self.dataset.batch(1).make_one_shot_iterator().get_next()
            n_samples = 0
            while True:
                try:
                    self.session.run([counter_iter])
                    n_samples+=1
                except tf.errors.OutOfRangeError:
                    # We reached the end of the dataset, break the loop and start a new epoch
                    self.dataset_size = n_samples
                    break
        remainder = np.remainder(self.dataset_size,self.batch_size)
        print("Ignoring {} samples, remainder of {} samples with a batch size of {}.".format(remainder, self.dataset_size, self.batch_size))
        self.dataset = self.dataset.skip(remainder)
        self.dataset = self.dataset.batch(self.batch_size)

        iterator = self.dataset.make_initializable_iterator()

        return iterator

    def initialize_and_restore(self):
        # Initialize all the variables
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        #Trying to load a checkpoint
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            self.checkpoint_counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.checkpoint_counter = 0
            print(" No checkpoints found. Starting a new net")

    # TODO: This is a try for enforcing the net to generate only the desired coding.
    def encoding_error(self):
        """
        Gets a float sample in range [-1;1] and outputs a map of the error between 0 and 1 representing how much each pixel
         is far from the encoding used
        :return:
        """
        pass


    def train(self, config):
        # Define an optimizer
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss_d, var_list=self.vars_d)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss_g, var_list=self.vars_g)

        # Generate the summaries
        summary_d, summary_g, summary_samples, writer = self.generate_summary()

        # Load and initialize the network
        self.initialize_and_restore()

        # Load the dataset
        dataset_iterator = self.load_dataset()

        i_epoch = 1
        for i_epoch in range(1, config.epoch+1):
            self.session.run([dataset_iterator.initializer])
            next_batch = dataset_iterator.get_next()
            batch_index = 0
            while True:
                # Train Step
                try:

                    train_batch = self.session.run(next_batch) # Batch of true samples
                    z_batch = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32) # Batch of noise

                    # D update
                    d, sum_d = self.session.run([d_optim, summary_d], feed_dict={self.x: train_batch['image'], self.z: z_batch})

                    # G Update (twice as stated in DCGAN comment, it makes sure d_loss does not go to zero
                    self.session.run([g_optim], feed_dict={self.z: z_batch})
                    g, sum_g = self.session.run([self.G, summary_g], feed_dict={self.z: z_batch})

                    # Write the summaries and increment the counter
                    writer.add_summary(sum_d, global_step=self.checkpoint_counter)
                    writer.add_summary(sum_g, global_step=self.checkpoint_counter)

                    batch_index += 1
                    self.checkpoint_counter += 1
                    print("Batch {}, Epoch {} of {}".format(batch_index, i_epoch, config.epoch))

                    # Check if the net should be saved
                    if np.mod(self.checkpoint_counter, self.save_net_every) == 2:
                        self.save(config.checkpoint_dir)
                        # Sample the network
                        np.random.seed(42)
                        z_sample = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                        samples = self.session.run([summary_samples], feed_dict={self.x: train_batch['image'], self.z: z_sample})
                        writer.add_summary(samples[0], global_step=self.checkpoint_counter)

                except tf.errors.OutOfRangeError:
                    # We reached the end of the dataset, break the loop and start a new epoch
                    i_epoch += 1
                    break


    def __init__(self, session, config):
        self.session = session
        self.dataset_path = config.dataset_path
        self.output_size = [config.height, config.width]
        self.output_channels = config.output_channels
        self.g_filter_depth = config.g_filter_depth
        self.d_filter_depth = config.d_filter_depth
        self.z_dim = config.z_dim
        self.batch_size= config.batch_size
        self.summary_folder= config.summary_folder
        self.checkpoint_dir = config.checkpoint_dir
        self.save_net_every = config.save_net_every
        self.dataset_size = config.dataset_size
        self.normalize_input = config.normalize_input
        self.build()

        pass

    def generate_sample_summary(self, sample_names):
        sample_summaries = [visualize_samples(name=name, input=self.G) for name in sample_names]
        merged_summaries = [tf.summary.merge([s]) for s in sample_summaries]
        return merged_summaries, tf.summary.FileWriter(self.summary_folder)

    def sample(self, seeds):

        # Load and initialize the network
        self.initialize_and_restore()

        names = ['seed_{}'.format(s) for s in seeds]
        summaries, writer = self.generate_sample_summary(names)

        for summary, seed in zip(summaries, seeds):
            np.random.seed(seed)
            z_sample = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            sample = self.session.run([summary], feed_dict={self.z: z_sample})
            writer.add_summary(sample[0], global_step=0)



if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_string("dataset_path", None, "Path to the .TfRecords file containing the dataset")
    flags.DEFINE_integer("dataset_size", None, "Number of samples contained in the .tfrecords dataset")
    flags.DEFINE_integer("height", 512, "Target sample height")
    flags.DEFINE_integer("width", 512, "Target sample width")
    flags.DEFINE_integer("output_channels", 3, "Target sample channels")
    flags.DEFINE_integer("g_filter_depth", 64, "number of filters for the first G convolution layer")
    flags.DEFINE_integer("d_filter_depth", 64, "number of filters for the first G convolution layer")
    flags.DEFINE_integer("z_dim", 100, "Dimension for the noise vector in input to G [100]")
    flags.DEFINE_integer("batch_size", 64, "Batch size")
    flags.DEFINE_integer("save_net_every", 5, "Number of train batches after which the next is saved")
    flags.DEFINE_boolean("normalize_input", True, "Whether to normalize input in range [0,1], Set to false if input is already normalized.")
    flags.DEFINE_boolean("train", True, "enable training if true, sample the net if false")
    flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("summary_folder", "/tmp/tflow/train", "Directory name to save the temporary files for visualization [/tmp/tflow/train]")


    FLAGS=flags.FLAGS

    with tf.Session() as s:
        gan = DoomGAN(session=s,
                      config=FLAGS
                      )
        show_all_variables()
        gan.train(FLAGS) if FLAGS.train else gan.sample(seeds=[42, 314, 123123, 65847968, 46546868])
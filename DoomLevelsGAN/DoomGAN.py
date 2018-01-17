import os

import numpy as np
import tensorflow.contrib as contrib
import pickle
import DoomLevelsGAN.DataTransform as DataTransform
from DoomDataset import DoomDataset
from DoomLevelsGAN.NNHelpers import *
import DoomLevelsGAN.inception.inception_score as inception
from scipy.spatial import cKDTree



class DoomGAN(object):
    def generator_generalized(self, z, hidden_layers, y=None):
        '''
        Parametrized version of the DCGAN generator, accept a variable number of hidden layers
        :param z: Noise as input to the net
        :param hidden_layers: The number of hidden layer to use. E.g. "5" will produce layers from h0 to h5 (output)
        :param stride: Convolutional stride to use for all the layers (tuple), default: (2,2)
        :return:
        '''

        def calc_filter_size(output_size, stride):
            return [int(math.ceil(float(z) / float(t))) for z, t in zip(output_size, stride)]

        def g_activation(x, name='g_a'):
            '''
            Activation function used in the generator, also includes batch/layer normalization if the architecture
            requires it
             '''
            norm_layer = batch_norm(name=name)
            return tf.nn.relu(norm_layer(x))

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
                size = [self.config.batch_size, g_size_filter[layer_id][0], g_size_filter[layer_id][1], layer['n_filters']]
                # First and last layer differs from the others
                if layer_id == 0:
                    size[0] = -1
                if layer_id == len(hidden_layers) - 1:
                    size[-1] = self.output_channels
                else:
                    # For everyone except the last layer, add the feature size
                    size[-1] += len(self.features)
                    pass
                g_size.append(size)

            # Concatenating the features to the input noise
            z = tf.concat([z, y], axis=1) if self.use_features else z

            # Size for the Z projection
            g_size_z_p = g_size[0][1] * g_size[0][2] * g_size[0][3]
            # Projection of Z
            z_p = linear_layer(z, g_size_z_p, 'g_h0_lin')
            # z_p = tf.concat([z_p, y], axis=1) if self.use_features else z_p

            self.layers_G = []
            self.w_G = []  # Keep track of the weights for visualization
            for layer_id, layer in enumerate(hidden_layers):
                if layer_id == 0:
                    l = g_activation(tf.reshape(z_p, g_size[0]))
                    # Concatenating the features to the activations
                    l = concatenate_features(l, y) if self.use_features else l
                else:
                    if layer_id == len(hidden_layers) - 1:
                        l, w = conv2d_transposed(self.layers_G[layer_id - 1], g_size[layer_id],
                                                 name='g_h{}'.format(layer_id),
                                                 stride_h=layer['stride'][0], stride_w=layer['stride'][1],
                                                 k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                                 remove_artifacts=layer['remove_artifacts'],
                                                 with_w=True
                                                 )
                        # We don't concatenate features at the last layer
                    else:
                        c, w = conv2d_transposed(self.layers_G[layer_id - 1], g_size[layer_id],
                                                 name='g_h{}'.format(layer_id),
                                                 stride_h=layer['stride'][0], stride_w=layer['stride'][1],
                                                 k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                                 remove_artifacts=layer['remove_artifacts'],
                                                 with_w=True)
                        l = g_activation(c, name='g_a{}'.format(layer_id))
                        l = concatenate_features(l, y) if self.use_features else l
                    self.w_G.append(w)
                self.layers_G.append(l)

        # return tf.nn.tanh(self.layers_G[-1]) # for the nature of the images it may be more convenient the range [0;1]
        return tf.nn.sigmoid(self.layers_G[-1])

    def discriminator_generalized(self, input, hidden_layers, reuse=False, y=None):
        def d_activation(x, name="d_a"):
            """
            Activation function for the discriminator. Also applies batch/layer normalization as requested by the architecture.
            :param x:
            :param name:
            :return:
            """
            if self.config.use_gradient_penalty:
                # WGAN_GP uses layer normalization instead of batch norm in the discriminator (critic)
                norm_layer = layer_norm(name=name)
            else:
                norm_layer = batch_norm(name=name)
            return leaky_relu(norm_layer(x))

        with tf.variable_scope("D") as scope:
            if reuse:
                scope.reuse_variables()
            layers_D = []
            w_D = []
            for layer_id, layer in enumerate(hidden_layers):
                if layer_id == 0:  # First layer (input)
                    input = concatenate_features(input, y)
                    c, w = conv2d(input, layer['n_filters'], name='d_a{}'.format(layer_id),
                                  k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                  stride_h=layer['stride'][0], stride_w=layer['stride'][1], with_w=True)
                    l = leaky_relu(c)
                else:
                    if layer_id == len(hidden_layers) - 1:  # Last layer (output)
                        l = linear_layer(tf.reshape(layers_D[-1], [self.config.batch_size, -1]), 1,
                                         scope='d_a{}'.format(layer_id))
                    else:  # Hidden layers
                        c, w = conv2d(layers_D[-1], layer['n_filters'], name="d_h{}".format(layer_id),
                                      k_h=layer['kernel_size'][0], k_w=layer['kernel_size'][1],
                                      stride_h=layer['stride'][0], stride_w=layer['stride'][1], with_w=True)
                        l = d_activation(c, name="g_a{}".format(layer_id))
                        l = concatenate_features(l, y) if self.use_features else l

                layers_D.append(l)
                w_D.append(w)
            if reuse:
                self.layers_D_fake = layers_D
                self.w_D_fake = w_D
            else:
                self.layers_D_real = layers_D
                self.w_D_real = w_D
        return tf.nn.sigmoid(layers_D[-1]), layers_D[-1]

    def loss_function(self):
        if self.config.use_wgan:
            self.loss_d = tf.reduce_mean(self.D_logits_fake - self.D_logits_real)
            self.loss_g_wgan = tf.reduce_mean(-self.D_logits_fake)

            if self.config.use_gradient_penalty:
                # Code from WGAN-GP paper https://arxiv.org/pdf/1704.00028.pdf (https://github.com/igul222/improved_wgan_training/)
                # Code has been modified to make one computation for the whole batch instead of repeating it for each sample
                alpha = tf.random_uniform(
                    shape=[self.config.batch_size, 1, 1, 1], # If multi-gpu, batchsize should be divided by num_gpu
                    minval=0.,
                    maxval=1.
                )
                alpha = tf.tile(alpha, multiples=[1, self.output_size[0], self.output_size[1], self.output_channels])
                differences = self.G - self.x_norm
                interpolates = self.x_norm + (alpha * differences)
                d_interp, d_logit_interp = self.discriminator_generalized(interpolates, self.d_layers, reuse=True, y=self.y_norm)
                gradients = tf.gradients(d_logit_interp, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                self.loss_d += self.config.lambda_gradient_penalty * gradient_penalty

            return self.loss_d, self.loss_g_wgan
        else:
            # Here we are using standard DCGAN loss
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

            self.balance = tf.abs(self.loss_g - self.loss_d)
            return self.loss_d, self.loss_g

    def build(self):
        # x: True inputs coming from the dataset
        # z: Noise in input to the generator
        # y: Feature vector


        # The channel are encoded dicrectly as in the dataset
        self.x = tf.placeholder(tf.float32, [self.config.batch_size] + self.output_size + [self.output_channels],
                                name="real_inputs")
        self.x_rotation = tf.placeholder(tf.float32, shape=(1), name="x_rotation")
        self.x_norm = DataTransform.scaling_maps(self.x, self.maps, self.config.dataset_path, self.config.use_sigmoid)
        self.x_norm = contrib.image.rotate(self.x_norm, self.x_rotation, "NEAREST")



        if self.use_features:
            self.y = tf.placeholder(tf.float32, [self.config.batch_size, len(self.features)])
            self.y_norm = DataTransform.scaling_features(self.y, self.features, self.config.dataset_path, self.config.use_sigmoid)

        self.z = tf.placeholder(tf.float32, [self.config.batch_size, self.config.z_dim], name='z')


        self.G = self.generator_generalized(self.z, hidden_layers=self.g_layers, y=self.y_norm)
        # G outputs in the same scale of x_norm. For reading output samples we can rescale them back to their original range
        self.G_rescaled = DataTransform.scaling_maps_inverse(self.G, self.maps, self.config.dataset_path, self.config.use_sigmoid)

        # Discriminator networks for each input type (real and generated)
        self.D_real, self.D_logits_real = self.discriminator_generalized(self.x_norm, self.d_layers, reuse=False,
                                                                         y=self.y_norm)
        self.D_fake, self.D_logits_fake = self.discriminator_generalized(self.G, self.d_layers, reuse=True, y=self.y_norm)

        # Define the loss function
        self.loss_d, self.loss_g = self.loss_function()

        # Collect the trainable variables for the optimizer
        vars = tf.trainable_variables()
        self.vars_d = [var for var in vars if 'd_' in var.name]
        self.vars_g = [var for var in vars if 'g_' in var.name]

    def generate_summary(self):
        # Pick a random sample from G and x and shows D activations
        s_g = visualize_samples('generator_output_rescaled', self.G_rescaled)
        s_x_norm = visualize_samples('true_input', self.x_norm)

        if self.config.use_wgan:
            s_loss_d = tf.summary.scalar('critic_loss', -tf.reduce_mean(self.loss_d))
            s_loss_g = tf.summary.scalar('generator_loss', tf.reduce_mean(self.loss_g_wgan))

            summary_d = tf.summary.merge([s_loss_d])
            summary_g = tf.summary.merge([s_loss_g])
            summary_out = tf.summary.merge([s_g, s_x_norm])
            return summary_d, summary_g, summary_out
        else:
            s_loss_d_real = tf.summary.scalar('d_loss_real_inputs', self.loss_d_real)
            s_loss_d_fake = tf.summary.scalar('d_loss_fake_inputs', self.loss_d_fake)
            s_loss_d = tf.summary.scalar('d_loss', self.loss_d)
            s_loss_g = tf.summary.scalar('g_loss', self.loss_g)

            s_d = tf.summary.merge([s_loss_d_real, s_loss_d_fake, s_loss_d])
            s_g = tf.summary.merge([s_loss_g])
            s_samples = tf.summary.merge([s_g])
            return s_d, s_g, s_samples

        # Code for showing one particular sample and activations
        # sample_index = 5
        # d_layers_to_show_g = self.layers_D_fake[1:-1]
        # d_layers_to_show_x = self.layers_D_real[1:-1]
        # s_g_chosen_input = visualize_samples('g_sample_{}'.format(sample_index), tf.slice(self.G, begin=[sample_index, 0, 0, 0], size=[1, -1, -1, -1]))
        # s_d_activations_g = visualize_activations('g_sample_{}_d_layer_'.format(sample_index), d_layers_to_show_g, sample_index)
        # s_x_chosen_input = visualize_samples('x_sample_{}'.format(sample_index), tf.slice(self.x_norm, begin=[sample_index, 0, 0, 0], size=[1, -1, -1, -1]))
        # s_d_activations_x = visualize_activations('x_sample_{}_d_layer_'.format(sample_index), d_layers_to_show_x, sample_index)


    def generate_metrics_summary(self):
        # Creating placeholders to feed numpy-evaluated metrics into tensorboard
        self.metrics = dict()
        for mapname in self.maps:
            self.metrics["entropy_mae_{}".format(mapname)] = tf.placeholder(tf.float32)
            self.metrics["similarity_mae_{}".format(mapname)] = tf.placeholder(tf.float32)
            self.metrics["encoding_error_{}".format(mapname)] = tf.placeholder(tf.float32)
        self.metrics["entropy_mae"] = tf.placeholder(tf.float32)
        self.metrics["similarity_mae"] = tf.placeholder(tf.float32)
        self.metrics["encoding_error"] = tf.placeholder(tf.float32)
        self.metrics["floor_corner_error"] = tf.placeholder(tf.float32)
        self.metrics["walls_corner_error"] = tf.placeholder(tf.float32)


        # Creating summaries for the metrics
        summaries = []
        for met in self.metrics.keys():
            summaries.append(tf.summary.scalar(met, self.metrics[met]))
        return tf.summary.merge(summaries)





    def save(self, checkpoint_dir):
        # Code from https://github.com/carpedm20/DCGAN-tensorflow
        model_name = "DOOMGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.config.checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=self.checkpoint_counter)

    def load(self, checkpoint_dir):
        # Code from https://github.com/carpedm20/DCGAN-tensorflow
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir+'checkpoint', ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_dataset(self, batch_size=None):
        """
        Loads both training and validation .TFRecords datasets from self.config.dataset_path pointing to a .meta file.
        :return: A tuple of dataset iterators (Training, Validation)
        """
        # Building paths
        assert os.path.isfile(self.config.dataset_path), "Dataset .meta not found at {}".format(self.config.dataset_path)
        train_path = ''.join(self.config.dataset_path.split('.meta')[:-1])+'-train.TFRecord'
        validation_path = ''.join(self.config.dataset_path.split('.meta')[:-1])+'-validation.TFRecord'
        assert os.path.isfile(train_path), "Dataset not found at {}. Check your file paths and try again".format(train_path)

        batch_size = batch_size or self.config.batch_size

        train_set = DoomDataset().read_from_TFRecords(train_path, target_size=self.output_size)
        train_set_size, validation_set_size = DoomDataset().get_dataset_count(self.config.dataset_path)
        
        train_remainder = np.remainder(train_set_size, batch_size)
        print(
            "Ignoring {} samples, remainder of {} samples with a batch size of {}.".format(train_remainder, train_set_size,
                                                                                           batch_size))
        train_set = train_set.shuffle(buffer_size=train_set_size*100)
        train_set = train_set.skip(train_remainder)
        train_set = train_set.shuffle(buffer_size=(train_set_size-train_remainder)*100)
        train_set = train_set.batch(batch_size)
        train_iter = train_set.make_initializable_iterator()

        if os.path.isfile(validation_path):
            validation_set = DoomDataset().read_from_TFRecords(validation_path, target_size=self.output_size)
            validation_remainder = np.remainder(validation_set_size, batch_size)
            print(
                "Ignoring {} samples, remainder of {} samples with a batch size of {}.".format(validation_remainder,
                                                                                               validation_set_size,
                                                                                               batch_size))
            validation_set = validation_set.shuffle(buffer_size=validation_set_size * 100)
            validation_set = validation_set.skip(validation_remainder)
            validation_set = validation_set.shuffle(buffer_size=(validation_set_size - validation_remainder) * 100)
            validation_set = validation_set.batch(batch_size)
            validation_iter = validation_set.make_initializable_iterator()

            return train_iter, validation_iter

        else:
            print("Validation dataset not found.")
            return train_iter

    def initialize_and_restore(self):
        # Initialize all the variables
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # Trying to load a checkpoint
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.config.checkpoint_dir)
        if could_load:
            self.checkpoint_counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.checkpoint_counter = 0
            print(" No checkpoints found. Starting a new net")

    def train(self, config):

        # OPTIMIZER DEFINITION
        if self.config.use_wgan:

            if not self.config.use_gradient_penalty:
                # Here we are using WGAN
                # Clipping D Weights
                clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.vars_d]
                # Define an optimizer
                d_optim = (tf.train.RMSPropOptimizer(learning_rate=config.wgan_lr)
                           .minimize(self.loss_d, var_list=self.vars_d))
                g_optim = (tf.train.RMSPropOptimizer(learning_rate=config.wgan_lr)
                           .minimize(self.loss_g_wgan, var_list=self.vars_g))
            else:
                # Here we are using WGAN-GP
                d_optim = tf.train.AdamOptimizer(config.wgangp_lr,
                                                 beta1=config.wgangp_beta1,
                                                 beta2=config.wgangp_beta2).minimize(self.loss_d,
                                                                                     var_list=self.vars_d)
                g_optim = tf.train.AdamOptimizer(config.wgangp_lr,
                                                 beta1=config.wgangp_beta1,
                                                 beta2=config.wgangp_beta2).minimize(self.loss_g,
                                                                                     var_list=self.vars_g)
        else:
            # Define an optimizer
            d_optim = tf.train.AdamOptimizer(config.dcgan_lr, beta1=config.dcgan_beta1).minimize(self.loss_d,
                                                                                                var_list=self.vars_d)
            g_optim = tf.train.AdamOptimizer(config.dcgan_lr, beta1=config.dcgan_beta1).minimize(self.loss_g,
                                                                                                var_list=self.vars_g)

        # SUMMARY GENERATION FOR TENSORBOARD VISUALIZATION
        # Summary for discriminator/critic loss, generator loss, generator output
        summary_d, summary_g, summary_samples = self.generate_summary()
        # Summary for validation metrics visualization
        summary_metrics = self.generate_metrics_summary()
        # Defining a writer for each run we are performing
        writer_train = tf.summary.FileWriter(self.config.summary_folder + 'train/')
        writer_valid = tf.summary.FileWriter(self.config.summary_folder + 'validation/')
        writer_ref = tf.summary.FileWriter(self.config.summary_folder + 'reference_sample/')



        # Load and initialize the network
        self.initialize_and_restore()

        # Load the dataset
        train_set_iter, valid_set_iter = self.load_dataset()

        # Define how many times to train d for each g step
        if self.config.use_wgan:
            d_iters = 5
        else:
            d_iters = 1

        for i_epoch in range(1, config.epoch + 1):
            self.session.run([train_set_iter.initializer])
            next_train_batch = train_set_iter.get_next()

            # Rotate the input for each epoch
            for rotation in [0, 90, 180, 270]:
                while True:
                    # Train Step
                    try:

                        # Train D
                        for i in range(d_iters):
                            # Get a new batch
                            train_batch = self.session.run(next_train_batch)  # Batch of true samples
                            z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                                np.float32)  # Batch of noise
                            x_batch = np.stack([train_batch[m] for m in maps], axis=-1)
                            y_batch = np.stack([train_batch[f] for f in self.features],
                                               axis=-1) if self.use_features else None
                            # Train D
                            self.session.run([d_optim], feed_dict={self.x: x_batch, self.y: y_batch, self.z: z_batch,
                                                                   self.x_rotation: [math.radians(rotation)]})
                            # Train G
                            if i == 0:
                                self.session.run([g_optim], feed_dict={self.y: y_batch, self.z: z_batch})

                        # Check if the net should be saved
                        if np.mod(self.checkpoint_counter, self.config.save_net_every) == 0 and self.checkpoint_counter > 100:
                            #self.save(config.checkpoint_dir)

                            # Calculating training loss
                            sum_d_train, sum_g_train = self.session.run([summary_d, summary_g],
                                                            feed_dict={self.x: x_batch, self.y: y_batch, self.z: z_batch,
                                                                       self.x_rotation: [math.radians(rotation)]})

                            # Validation step
                            self.session.run([valid_set_iter.initializer])
                            next_valid_batch = valid_set_iter.get_next()

                            validation_batch = self.session.run(next_valid_batch)
                            y_val_batch = np.stack([validation_batch[f] for f in self.features],
                                               axis=-1) if self.use_features else None
                            # "True" levels to evaluate against
                            x_val_batch = np.stack([validation_batch[m] for m in maps], axis=-1)
                            z_val_batch = np.random.uniform(-1, 1,
                                                        [self.config.batch_size, self.config.z_dim]).astype(
                                np.float32)
                            g_val_batch = self.session.run([self.G_rescaled],
                                                      feed_dict={self.z: z_val_batch, self.y: y_val_batch})[0]

                            # Calculating validation loss for critic and generator
                            sum_d_valid, sum_g_valid, sum_out_valid = self.session.run([summary_d, summary_g, summary_samples],
                                             feed_dict={self.x: x_val_batch, self.y: y_val_batch, self.z: z_val_batch,
                                                        self.x_rotation: [math.radians(0)]})

                            # "offline" metrics calculation (using numpy, then sending the results to tensorboard)
                            metric_results = self.evaluate(g_val_batch, x_val_batch)
                            val_feed_dict = {self.metrics[metric]: metric_results[metric] for metric in self.metrics.keys()}
                            sum_metrics_valid = self.session.run([summary_metrics], feed_dict=val_feed_dict)[0]

                            # REFERENCE SAMPLE PLOTTING
                            # A reference sample is kept frozen and shown at each validation step to visually understand
                            # how the training is proceeding
                            x_ref, y_ref, z_ref = self.get_reference_sample(x_val_batch, y_val_batch, z_val_batch)
                            sum_out_ref, g_ref = self.session.run(
                                [summary_samples, self.G_rescaled],
                                feed_dict={self.x: x_ref, self.y: y_ref, self.z: z_ref,
                                           self.x_rotation: [math.radians(0)]})
                            ref_metric_results = self.evaluate(g_ref, x_ref)
                            ref_feed_dict = {self.metrics[metric]: ref_metric_results[metric] for metric in self.metrics.keys()}
                            sum_metrics_ref = self.session.run([summary_metrics], feed_dict=ref_feed_dict)[0]

                            # Writing the summary for the validation run
                            writer_valid.add_summary(sum_d_valid, global_step=self.checkpoint_counter)
                            writer_valid.add_summary(sum_g_valid, global_step=self.checkpoint_counter)
                            writer_valid.add_summary(sum_out_valid, global_step=self.checkpoint_counter)
                            writer_valid.add_summary(sum_metrics_valid, global_step=self.checkpoint_counter)
                            writer_ref.add_summary(sum_out_ref, global_step=self.checkpoint_counter)
                            writer_ref.add_summary(sum_metrics_ref, global_step=self.checkpoint_counter)
                            # Writing the summary for the train run
                            writer_train.add_summary(sum_d_train, global_step=self.checkpoint_counter)
                            writer_train.add_summary(sum_g_train, global_step=self.checkpoint_counter)

                        # incrementing the iteration counter
                        self.checkpoint_counter += 1
                        print("Iteration: {}".format(self.checkpoint_counter))

                    except tf.errors.OutOfRangeError:
                        # We reached the end of the dataset, break the loop and start a new epoch
                        break
            i_epoch += 1

    def get_reference_sample(self, current_x, current_y, current_z):
        """
        Loads a saved reference batch used to visualize how the learning phase proceed on the same generated sample
        If no samples are found in the corresponding file, then the provided batches will be saved as reference sample
        :param current_x:
        :param current_y:
        :param current_z:
        :return: a tuple of x, y, z reference batches
        """
        if all([inp is not None for inp in self.reference_sample.values()]):
            return self.reference_sample['x'], self.reference_sample['y'], self.reference_sample['z']
        # Try loading the samples from file
        paths = ["reference_sample_{}.npy".format(inp) for inp in [self.reference_sample.keys()]]
        is_file = [os.path.isfile(p) for p in paths]
        if not all(is_file):
            # Save a new reference batch
            np.save("reference_sample_x.npy", current_x)
            np.save("reference_sample_y.npy", current_y)
            np.save("reference_sample_z.npy", current_z)
            self.reference_sample['x'] = current_x.copy()
            self.reference_sample['y'] = current_y.copy()
            self.reference_sample['z'] = current_z.copy()
        else:
            self.reference_sample['x'] = np.load("reference_sample_x.npy")
            self.reference_sample['y'] = np.load("reference_sample_y.npy")
            self.reference_sample['z'] = np.load("reference_sample_z.npy")
        return self.reference_sample['x'], self.reference_sample['y'], self.reference_sample['z']


    def __init__(self, session, config, features, maps, d_layers, g_layers):
        self.session = session
        self.d_layers = d_layers
        self.g_layers = g_layers
        self.config = config
        self.output_size = [config.height, config.width]
        self.features = features
        self.use_features = len(self.features) > 0
        self.maps = maps
        self.output_channels = len(maps)
        self.nearest_features_tree = None
        self.reference_sample = {'x': None, 'y': None, 'z':None}
        self.build()

    def sample(self, y_factors = None, y_batch = None, seed=None, freeze_z=False, postprocess=False, save=False):
        """
        Sample the network with a given generator input. Various options are available.

        :param y_factors: feature vector of shape (batch, features), each value in {-1; [0,1]} where -1 means "average value for this feature", while [0;1] corresponds to values that go from -std to +std for that feature.
        :param y_batch: direct batch of feature values to feed to the generator. Either y_factors or y_batch must be different from None.
        :param seed: Seed for input noise generation. If None the seed is random at each z sampling [None]
        :param freeze_z: if True use the same input noise z for each generated sample
        :param postprocess: If true, the generated levels are postprocessed (denoised and eventually rescaled) and returned as they would be passed to the WadEditor.
        :param save: Has no effect is postprocess is not True. [False]
                    - False: Levels are returned as numpy array and not saved anywhere
                    - 'PNG': Network output is saved to the generated_samples directory
                    - 'WAD': Levels are converted to .WAD and saved in the generated_samples directory along with a descriptive image of the level
        """

        assert (y_factors is not None) ^ (y_batch is not None)
        if y_factors is not None:
            # Build a y vector from the y_factors by finding the nearest neighbors in the dataset
            y_feature_space = DoomDataset().get_feature_sample(self.config.dataset_path, y_factors, features, 'std')
            if self.nearest_features_tree is None:
                self.initialize_and_restore()
                loaded_features = DoomDataset().load_features(self.config.dataset_path, self.features, self.output_size)
                self.nearest_features_tree = cKDTree(loaded_features)

            y = np.zeros_like(y_feature_space)
            for s in range(self.config.batch_size):
                distance, nearest_index = self.nearest_features_tree.query(y_feature_space[s])
                y[s, :] = loaded_features[nearest_index]
        if y_batch:
            # the y_batch is provided externally
            y = y_batch
        if seed is not None:
            np.random.seed(seed)
        if freeze_z:
            z_sample = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(
            np.float32)
            z_batch = np.repeat(z_sample, self.config.batch_size, axis=0)
        else:
            z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                np.float32)
        result = self.session.run([self.G_rescaled],
                         feed_dict={self.z: z_batch, self.y: y})[0]
        if postprocess:
            if save is False:
                return DataTransform.postprocess_output(result, self.maps, folder=None)
            elif save == 'PNG':
                return DataTransform.postprocess_output(result, self.maps)
            elif save == 'WAD':
                return DataTransform.build_levels(result, self.maps, self.config.batch_size)
        return result


    def evaluate(self, s_gen, s_true):
        """
        Compute several metrics between a batch of generated sample and true sample corresponding on the same y vector.
        :param s_gen: the batch of generated samples
        :param s_true: the batch of true samples
        :return:
        """
        s_gen = s_gen.astype(np.uint8)
        # TODO: Complete this function
        from scipy.stats import entropy
        from skimage.measure import compare_ssim
        from DoomLevelsGAN.OutputEvaluation import encoding_error
        from skimage.feature import corner_harris, corner_peaks
        corner_error = lambda x, y: np.power((np.power(x - y, 2) / (x * y)), 0.5)

        metrics = {}
        # Computing color-based histogram
        hist_gen = np.zeros(shape=(self.config.batch_size, len(self.maps), 255))
        hist_tru = np.zeros(shape=(self.config.batch_size, len(self.maps), 255))
        entropies = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        ssim = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        enc_error = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        floor_corner_error = np.zeros(shape=(self.config.batch_size, 1))
        walls_corner_error = np.zeros(shape=(self.config.batch_size, 1))


        for m, mname in enumerate(self.maps):
            for s in range(self.config.batch_size):
                h_gen = np.histogram(s_gen[s, :, :, m], bins=255, range=(0, 255), density=True)[0]
                h_tru = np.histogram(s_true[s, :, :, m], bins=255, range=(0, 255), density=True)[0]
                e = entropy(h_gen)-entropy(h_tru)
                hist_gen[s, m, :] = h_gen
                hist_tru[s, m, :] = h_tru
                entropies[s, m, :] = e
                ssim[s, m, :] = compare_ssim(s_gen[s, :, :, m], s_true[s, :, :, m])

                # TODO: Add encoding errors
                if mname in ['floormap','wallmap']:
                    enc_error[s,m,:] = np.mean(encoding_error(s_gen[s,:,:,m], 255))
                elif mname in ['heightmap', 'thingsmap', 'triggermap']:
                    enc_error[s,m,:] = np.mean(encoding_error(s_gen[s, :, :, m], 1))
                if mname == 'floormap':
                    corners_floor_true = len(corner_peaks(corner_harris(s_true[s, :,:,m])))
                    corners_floor_gen = len(corner_peaks(corner_harris(s_gen[s, :,:,m])))
                    floor_corner_error[s,:] = corner_error(corners_floor_true, corners_floor_gen)
                if mname == 'wallmap':
                    corners_walls_true = len(corner_peaks(corner_harris(s_true[s, :,:,m])))
                    corners_walls_gen = len(corner_peaks(corner_harris(s_gen[s, :,:,m])))
                    walls_corner_error[s,:] = corner_error(corners_walls_true, corners_walls_gen)

            metrics["entropy_mae_{}".format(mname)] = np.mean(entropies[:,m,:])
            metrics["similarity_mae_{}".format(mname)] = np.mean(ssim[:,m,:])
            metrics["encoding_error_{}".format(mname)] = np.mean(enc_error[:,m,:])
        metrics["entropy_mae"] = np.mean(entropies)
        metrics["similarity_mae"] = np.mean(ssim)
        metrics["encoding_error"] = np.mean(enc_error)
        metrics["floor_corner_error"] = np.mean(floor_corner_error)
        metrics["walls_corner_error"] =  np.mean(walls_corner_error)

        return metrics

    def generate_levels_feature_interpolation(self, feature_to_interpolate, seed=None):
        # TODO: Implement this or remove it
        def slerp(val, low, high):
            """Spherical interpolation. val has a range of 0 to 1.
            Code from: https://github.com/dribnet/plat
            """
            if val <= 0:
                return low
            elif val >= 1:
                return high
            elif np.allclose(low, high):
                return low

            dot = np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high))
            omega = np.arccos(dot)
            so = np.sin(omega)
            return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

        # Load and initialize the network
        self.initialize_and_restore()
        # Fix a sample so the change gets more visible
        if seed is not None:
            np.random.seed(seed)
        z_sample = np.random.normal(0, 1, [self.config.z_dim]).astype(np.float32)
        # Replicate the same noise vector for each sample (each level should be the same except for the controlled feature)
        z_sample = np.tile(z_sample, (self.config.batch_size, 1))
        feat_factors = np.ones(shape=(self.config.batch_size, len(self.features)))*-1
        # building a matrix factors of size (batch_size, len(features)).
        # The factors will all be -1 (so we get the mean value for every feature) except the column
        # corresponding to the feature that has to be interpolated, which will range from min to max.
        for f, fname in enumerate(self.features):
            if fname == feature_to_interpolate:
                # TODO: Still using linear interpolation, spherical may lead to better results (between which starting points?)
                feat_factors[:, f] = np.linspace(0,1,num=self.config.batch_size)
        y = DoomDataset().get_feature_sample(self.config.dataset_path, feat_factors, features=self.features, extremes='minmax')
        # Constructing a y vector with a different y for each sample in the batch
        samples = self.session.run([self.G_rescaled], feed_dict={self.z: z_sample, self.y: y})
        samples = DataTransform.postprocess_output(samples[0], self.maps, folder=None)
        DataTransform.build_levels(samples, self.maps, self.config.batch_size, call_node_builder=False,
                                   level_images_path='./interpolated_features/{}/'.format(feature_to_interpolate))




    def inception_score(self):
        # TODO: Inception score seems to be not so meaningful in this case
        scores = {mapname: list() for mapname in self.maps}
        # Load the dataset
        dataset_iterator = self.load_dataset()

        self.session.run([dataset_iterator.initializer])
        next_batch = dataset_iterator.get_next()
        batch_index = 0
        while True:
            # Train Step
            try:

                train_batch = self.session.run(next_batch)  # Batch of true samples
                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                    np.float32)  # Batch of noise
                x_batch = np.stack([train_batch[m] for m in maps], axis=-1)
                y_batch = np.stack([train_batch[f] for f in self.features], axis=-1) if self.use_features else None
                # Transpose from (batch, width, height, map) to (map, batch, width, height, 1)
                x_batch = np.expand_dims(x_batch.transpose(-1, 0, 1, 2), axis=-1)
                # now replicate the last channel to make it rgb (inception works on rgb)
                x_batch = np.repeat(x_batch, 3, -1)
                for id, mapname in enumerate(self.maps):
                    x_map = list(x_batch[id])
                    scores[mapname] = inception.get_inception_score(x_map)


            except tf.errors.OutOfRangeError:
                # We reached the end of the dataset, break the loop and start a new epoch
                break
        print("Done")

    def nearest_neighbor_score(self, calc_entropy=False):
        # TODO: This should be done on an held-out set
        import sklearn.model_selection
        from sklearn.neighbors import KNeighborsClassifier
        # Load and initialize the network
        self.initialize_and_restore()
        # Go through a full epoch to load all the data and generate relative samples

        # Load the dataset
        dataset_iterator = self.load_dataset()

        samples = {'x': list(), 'y':list(), 'g':list()}

        self.session.run([dataset_iterator.initializer])
        next_batch = dataset_iterator.get_next()
        x_incep = list()
        g_incep = list()
        entr = {m: {'x':list(), 'g':list()} for m in self.maps}
        while True:
            try:
                # Get a new batch
                train_batch = self.session.run(next_batch)  # Batch of true samples
                x_batch = np.stack([train_batch[m] for m in maps], axis=-1)
                y_batch = np.stack([train_batch[f] for f in self.features],
                                   axis=-1)
                # Normalizing the x_batch according to DoomGAN
                x_batch = self.session.run([self.x_norm], feed_dict={self.x: x_batch, self.x_rotation: [0]})[0]
                g_batch = self.sample(y_batch=y_batch)
                g_batch = np.around(g_batch).astype(np.uint8)
                x_batch = np.around(x_batch*255).astype(np.uint8)






                # Mean Entropy calculation
                if calc_entropy:
                    from skimage.filters.rank import entropy
                    from skimage.morphology import disk
                    print("Entropy calculation")
                    for m, mapname in enumerate(self.maps):
                        for s in range(self.config.batch_size):
                            x_entropy = np.mean(entropy(x_batch[s,:,:,m], disk(2)))
                            g_entropy = np.mean(entropy(g_batch[s,:,:,m], disk(2)))
                            entr[mapname]['x'].append(x_entropy)
                            entr[mapname]['g'].append(g_entropy)
                            print('.', end='')

            except tf.errors.OutOfRangeError:
                #  reached the end of the dataset
                break
        # Mean Entropy calculation
        if calc_entropy:
           for m, mapname in enumerate(self.maps):
               x_entropy = np.mean(entr[mapname]['x'])
               g_entropy = np.mean(entr[mapname]['g'])
               print("X mean entropy: {}".format(x_entropy))
               print("G mean entropy: {}".format(g_entropy))


        x_test = np.repeat(np.expand_dims(x_batch[:, :, :, 1], axis=-1), 3, axis=-1)
        g_test = np.repeat(np.expand_dims(g_batch[:, :, :, 1], axis=-1), 3, axis=-1)

        x_incep.append(list(inception.get_inception_score(list(x_test))))
        g_incep.append(list(inception.get_inception_score(list(g_test))))

        x_inc_mean = np.mean(np.asarray(x_incep)[:, 0])
        g_inc_mean = np.mean(np.asarray(g_incep)[:, 0])
        print("Calculating inception score for x and g sets (floormap only)")
        print("X: {} \n G: {} \n".format(x_inc_mean, g_inc_mean))
        # all_samples = np.reshape(all_samples, [all_samples.shape[0], -1])
        labels = np.concatenate((1*np.ones(len(samples['x'])), -1*np.ones(len(samples['g']))))


def clean_tensorboard_cache(tensorBoardPath):
    # Removing previous tensorboard session
    import shutil
    print('Cleaning temp tensorboard directory: {}'.format(tensorBoardPath))
    shutil.rmtree(tensorBoardPath, ignore_errors=True)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 400, "Epoch to train [400]")
    # HYPERPARAMETERS
    # DCGAN
    flags.DEFINE_float("dcgan_lr", 2e-4, "Learning rate for DCGAN adam [2e-4]")
    flags.DEFINE_float("dcgan_beta1", 0.5, "Momentum term of adam [0.5]")
    # WGAN
    flags.DEFINE_float("wgan_lr", 5e-5, "Learning rate for WGAN RMSProp [5e-5]")
    # WGAN-GP
    flags.DEFINE_float("wgangp_lr", 2e-4, "Learning rate for WGAN-GP adam [2e-4]")
    flags.DEFINE_float("wgangp_beta1", 0., "beta1 for WGAN-GP adam [0.]")
    flags.DEFINE_float("wgangp_beta2", 0.9, "beta2 for WGAN-GP adam [0.9]")
    flags.DEFINE_integer("lambda_gradient_penalty", 10, "Gradient penalty lambda hyperparameter [10]")


    flags.DEFINE_float("lr_wgangp", 5e-5, "Learning rate of for WGAN RMSProp [5e-5]")

    flags.DEFINE_float("beta1_wgan", 0, "Momentum term of adam [0.5]")
    flags.DEFINE_string("dataset_path", None, "Path to the .meta file of the corresponding .TFRecord datasets")
    flags.DEFINE_integer("height", 128, "Target sample height")
    flags.DEFINE_integer("width", 128, "Target sample width")
    flags.DEFINE_integer("z_dim", 100, "Dimension for the noise vector in input to G [100]")
    flags.DEFINE_integer("batch_size", 64, "Batch size")
    flags.DEFINE_integer("save_net_every", 100, "Number of train batches after which the next is saved")
    flags.DEFINE_boolean("train", True, "enable training if true")
    flags.DEFINE_boolean("generate", False, "If true, generate some levels with a fixed y value and seed and save them into ./generated_samples")
    flags.DEFINE_boolean("interpolate", False, "If true, generate levels by interpolating the feature vector along each dimension")
    flags.DEFINE_boolean("test", False, "If true, compute evaluation metrics over produced samples")
    flags.DEFINE_boolean("use_sigmoid", True,
                         "If true, uses sigmoid activations for G outputs, if false uses Tanh. Data will be normalized accordingly")
    flags.DEFINE_boolean("use_wgan", True, "Whether to use the Wesserstein GAN model or the standard GAN")
    flags.DEFINE_boolean("use_gradient_penalty", True, "Whether to use the gradient penalty from WGAN_GP architecture or the WGAN weight clipping")

    flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("summary_folder", "/tmp/tflow/",
                        "Directory name to save the temporary files for visualization [/tmp/tflow/]")

    FLAGS = flags.FLAGS

    with tf.Session() as s:
        #clean_tensorboard_cache('/tmp/tflow')
        # Define here which features to use
        features = ['height', 'width',
                   'number_of_sectors',
                   'sector_area_avg',
                   'sector_aspect_ratio_avg',
                   'lines_per_sector_avg',
                   'walkable_percentage',
                   'level_extent',
                   'level_solidity',
                   'artifacts_per_walkable_area',
                   'powerups_per_walkable_area',
                   'weapons_per_walkable_area',
                   'ammunitions_per_walkable_area',
                   'keys_per_walkable_area',
                   'monsters_per_walkable_area',
                   'obstacles_per_walkable_area',
                   'decorations_per_walkable_area']
        maps = ['floormap', 'heightmap', 'wallmap', 'thingsmap', 'triggermap']



        # Generator network
        g_layers = [
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 16, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 8, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 4, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 2, 'remove_artifacts': False},
        ]

        d_layers = [
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 2, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 4, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 8, 'remove_artifacts': False},
            {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 16, 'remove_artifacts': False},
        ]



        gan = DoomGAN(session=s,
                      config=FLAGS, features=features, maps=maps,
                      g_layers = g_layers, d_layers = d_layers
                      )
        show_all_variables()
        if FLAGS.train:
            gan.train(FLAGS)
        else:
            feat_factors = [-1 for f in features]
            if FLAGS.generate:
                #factors = np.tile(0.5, (FLAGS.batch_size, len(features)))
                factors = np.random.normal(0.5, 0.1, size=(FLAGS.batch_size, len(features)))
                #factors = np.repeat(np.random.uniform(0,1, (1, len(features))), FLAGS.batch_size, axis=0)
                #y = np.tile (np.asarray(y), (FLAGS.batch_size, 1))
                gan.sample(y_factors=factors, postprocess=True, save='PNG', seed=654654)
            if FLAGS.interpolate:
                for feat in features:
                    gan.generate_levels_feature_interpolation(feature_to_interpolate=feat, seed=123456789)
            if FLAGS.test:
                gan.evaluate()
                #gan.nearest_neighbor_score()

import matplotlib
#matplotlib.use('Agg') # This is for avoiding crashes for missing tkinter in Docker TODO: Enable this again
import os
import DoomLevelsGAN.network_architecture as architecture
import tensorflow.contrib as contrib
import DoomLevelsGAN.DataTransform as DataTransform
from DoomDataset import DoomDataset
from DoomLevelsGAN.NNHelpers import *
from WAD_Parser.RoomTopology import *
from scipy.spatial import cKDTree
from WAD_Parser.Dictionaries import Features as all_features



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
                    input = concatenate_features(input, y) if self.use_features else input
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




        self.y = tf.placeholder(tf.float32, [self.config.batch_size, len(self.features)]) if self.use_features else None
        self.y_norm = DataTransform.scaling_features(self.y, self.features, self.config.dataset_path, self.config.use_sigmoid) if self.use_features else None

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
            self.metrics["similarity_{}".format(mapname)] = tf.placeholder(tf.float32)
            self.metrics["encoding_error_{}".format(mapname)] = tf.placeholder(tf.float32)
        self.metrics["entropy_mae"] = tf.placeholder(tf.float32)
        self.metrics["similarity"] = tf.placeholder(tf.float32)
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
        #checkpoint_dir = os.path.join(checkpoint_dir, self.config.checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=self.checkpoint_counter)

    def load(self, checkpoint_dir):
        # Code from https://github.com/carpedm20/DCGAN-tensorflow
        import re
        print(" [*] Reading checkpoints...")

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

    def load_dataset(self, batch_size=None):
        """
        Loads both training and validation .TFRecords datasets from self.config.dataset_path pointing to a .meta file.
        :return: A tuple of dataset iterators (Training, Validation)
        """
        # Building paths
        assert os.path.isfile(self.config.dataset_path), "Dataset .meta not found at {}".format(self.config.dataset_path)
        train_path = DoomDataset().get_dataset_path(self.config.dataset_path, 'train')
        validation_path = DoomDataset().get_dataset_path(self.config.dataset_path, 'validation')
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
            tf.global_variables_initializer().run(session=self.session)
        except:
            tf.initialize_all_variables().run(session=self.session)

        # Trying to load a checkpoint
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.config.checkpoint_dir)
        if could_load:
            self.checkpoint_counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            self.checkpoint_counter = 0
            print(" No checkpoints found. Starting a new net")

    def inspect_data(self):
        import WAD_Parser.Dictionaries.Features as all_feature_list
        """ Loads the dataset for data inspection/visualization """
        # Load the dataset
        train_set_iter, valid_set_iter = self.load_dataset()
        self.session.run([train_set_iter.initializer])
        next_train_batch = train_set_iter.get_next()
        train_batch = self.session.run(next_train_batch)
        x_batch = np.stack([train_batch[m] for m in maps], axis=-1)
        all_feats = np.stack([train_batch[f] for f in all_feature_list.features], axis=-1)
        DoomDataset().show_sample_batch(x_batch,0)
        sel = [4, 6, 10, 11, 31]
        selected_samples = dict()
        for k in train_batch:
            selected_samples[k] = np.take(train_batch[k], sel, axis=0)


        DoomDataset().compare_features(selected_samples)
        print("Stop")


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

        # Load the datasets
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
                            net_inputs = {self.x: x_batch,
                                          self.z: z_batch,
                                          self.x_rotation: [math.radians(rotation)]}
                            if self.use_features:
                                net_inputs[self.y] = y_batch
                            self.session.run([d_optim], feed_dict=net_inputs)
                            # Train G
                            if i == 0:
                                net_inputs = {self.z: z_batch}
                                if self.use_features:
                                    net_inputs[self.y] = y_batch
                                self.session.run([g_optim], feed_dict=net_inputs)

                        # Check if the net should be saved
                        if np.mod(self.checkpoint_counter, self.config.save_net_every) == 0:
                            self.save(config.checkpoint_dir)

                            # Calculating training loss
                            net_inputs = {self.x: x_batch,
                                          self.z: z_batch,
                                          self.x_rotation: [math.radians(rotation)]}
                            if self.use_features:
                                net_inputs[self.y] = y_batch
                            sum_d_train, sum_g_train = self.session.run([summary_d, summary_g],
                                                            feed_dict=net_inputs)

                            # Validation step
                            self.session.run([valid_set_iter.initializer])
                            next_valid_batch = valid_set_iter.get_next()

                            validation_batch = self.session.run(next_valid_batch)
                            y_val_batch = np.stack([validation_batch[f] for f in self.features],
                                               axis=-1) if self.use_features else None
                            # "True" levels to compute_quality_metrics against
                            x_val_batch = np.stack([validation_batch[m] for m in maps], axis=-1)
                            z_val_batch = np.random.uniform(-1, 1,
                                                        [self.config.batch_size, self.config.z_dim]).astype(
                                np.float32)

                            net_inputs ={self.z: z_val_batch}
                            if self.use_features:
                                net_inputs[self.y] = y_val_batch
                            g_val_batch = self.session.run([self.G_rescaled],
                                                      feed_dict=net_inputs)[0]

                            # Calculating validation loss for critic and generator
                            net_inputs = {self.x: x_val_batch,
                                          self.z: z_val_batch,
                                          self.x_rotation: [math.radians(0)]}
                            if self.use_features:
                                net_inputs[self.y] = y_val_batch
                            sum_d_valid, sum_g_valid, sum_out_valid = self.session.run([summary_d, summary_g, summary_samples],
                                                                                       feed_dict=net_inputs)

                            # "offline" metrics calculation (using numpy, then sending the results to tensorboard)
                            metric_results = self.compute_quality_metrics(g_val_batch, x_val_batch)
                            val_feed_dict = {self.metrics[metric]: metric_results[metric] for metric in self.metrics.keys()}
                            sum_metrics_valid = self.session.run([summary_metrics], feed_dict=val_feed_dict)[0]

                            # REFERENCE SAMPLE PLOTTING
                            # A reference sample is kept frozen and shown at each validation step to visually understand
                            # how the training is proceeding
                            if self.use_features:
                                x_ref, y_ref, z_ref = self.get_reference_sample(x_val_batch, y_val_batch, z_val_batch)
                            else:
                                x_ref, z_ref = self.get_reference_sample(x_val_batch, None, z_val_batch)
                            net_inputs = {self.x: x_ref,
                                          self.z: z_ref,
                                          self.x_rotation: [math.radians(0)]}
                            if self.use_features:
                                net_inputs[self.y] = y_ref
                            sum_out_ref, g_ref = self.session.run([summary_samples, self.G_rescaled], feed_dict=net_inputs)
                            ref_metric_results = self.compute_quality_metrics(g_ref, x_ref)
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
        print("Training complete after {} epochs corresponding to {} iterations".format(i_epoch, self.checkpoint_counter))

    def get_reference_sample(self, current_x, current_y, current_z):
        """
        Loads a saved reference batch used to visualize how the learning phase proceed on the same generated sample
        If no samples are found in the corresponding file, then the provided batches will be saved as reference sample.
        If the model doen't use features, then the output corresponding to y will be None
        :param current_x:
        :param current_y:
        :param current_z:
        :return: a tuple of x, y, z reference batches if the network uses features, otherwise a tuple of x, z
        """
        if all([inp is not None for inp in self.reference_sample.values()]):
            if self.use_features:
                return self.reference_sample['x'], self.reference_sample['y'], self.reference_sample['z']
            else:
                return self.reference_sample['x'],self.reference_sample['z']
        # Try loading the samples from file
        paths = ["{}reference_sample_{}.npy".format(self.config.ref_sample_folder, inp) for inp in [self.reference_sample.keys()]]
        is_file = [os.path.isfile(p) for p in paths]
        if not all(is_file):
            # Save a new reference batch
            np.save("{}reference_sample_x.npy".format(self.config.ref_sample_folder), current_x)
            np.save("{}reference_sample_z.npy".format(self.config.ref_sample_folder), current_z)
            self.reference_sample['x'] = current_x.copy()
            self.reference_sample['z'] = current_z.copy()
            if self.use_features:
                np.save("{}reference_sample_y.npy".format(self.config.ref_sample_folder), current_y)
                self.reference_sample['y'] = current_y.copy()
        else:
            self.reference_sample['x'] = np.load("{}reference_sample_x.npy".format(self.config.ref_sample_folder))
            self.reference_sample['z'] = np.load("{}reference_sample_z.npy".format(self.config.ref_sample_folder))
        if self.use_features:
            self.reference_sample['y'] = np.load("{}reference_sample_y.npy".format(self.config.ref_sample_folder))
            return self.reference_sample['x'], self.reference_sample['y'], self.reference_sample['z']
        else:
            return self.reference_sample['x'], self.reference_sample['z']

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
        self.reference_sample = {'x': None, 'y': None, 'z':None} if self.use_features else {'x': None, 'z':None}
        self.build()

    def sample(self, mode, sample_from_dataset='validation', y_factors = None, y_batch = None, seed=None, freeze_z=False, z_override=None, postprocess=False, save=False, folder=None):
        """
        Samples the network with a given generator input. Various options for selecting the feature vector (y) and the noise vector (z) are possible.
        If the network model doesn't use any feature, than Y sampling is ignored.
        Y-Sampling has three main modes of operation for sampling the y vector, defined by the 'mode' parameter.
        1) 'dataset' means that the y vector is taken from random samples belonging to either the training or validation set (controlled by sample_from_dataset)
        2) 'factors' means that the y vector is controlled by the 'y_factors' input vector, for which each value can be '-1', corresponding to the average dataset value or in [0,1], meaning [-std, +std] of the corresponding feature.
        3) 'nearest' means that the y vector is sampled like in factors, but it's not used directly as it. It is instead used to find the nearest neighbour in the dataset and that value is used for sampling
        4) 'direct' means that the y vector is taken directly in input as the parameter y_batch and the sampling is made externally. This may lead to poor results if you don't sample the y vector in a meaningful way.


        :param sample_from_dataset: If 'validation' (default) or 'train', then y is randomly taken from samples belonging to the corresponding dataset
        :param y_factors: feature vector of shape (batch, features), each value in {-1; [0,1]} where -1 means "average value for this feature", while [0;1] corresponds to values that go from -std to +std for that feature. The nearest neighbour among the training data is picked as y input
        :param y_batch: direct batch of feature values to feed to the generator. Either y_factors or y_batch must be different from None if sample_for_dataset is None.
        :param seed: Seed for input noise generation. If None the seed is random at each z sampling [None]
        :param freeze_z: if True use the same input noise z for each generated sample in the batch
        :param z_override: if set to a numpy array, it is used instead of sampling a new z. If freeze_z is True, then it must have the shape of a single z sample.
        :param postprocess: If true, the generated levels are postprocessed (denoised and eventually rescaled) and returned as they would be passed to the WadEditor.
        :param save: Has no effect is postprocess is not True. [False]
                    - False: Levels are returned as numpy array and not saved anywhere
                    - 'PNG': Network output is saved to the generated_samples directory
                    - 'WAD': Levels are converted to .WAD and saved in the generated_samples directory along with a descriptive image of the level
        :param folder: Where to folder generated samples
        """
        x_true_batch = None # This is set only if Y is picked in 'dataset' mode and it's saved along with the generated samples
        # Y Sampling
        if self.use_features:
            assert mode in ['dataset', 'factors', 'direct', 'nearest'], "Mode can only be 'dataset', 'factors' or 'direct'"
            if mode == 'dataset':
                assert sample_from_dataset in ['train', 'validation'], "If mode is 'dataset' then 'sample_from_dataset' must be either 'train' or 'validation'"
                # Load the datasets
                # Load the dataset
                train_set_iter, valid_set_iter = self.load_dataset()
                chosen_iter = train_set_iter if sample_from_dataset == 'train' else valid_set_iter
                self.session.run([chosen_iter.initializer])
                next_batch = chosen_iter.get_next()
                batch = self.session.run(next_batch)
                x_true_batch = np.stack([batch[m] for m in self.maps], axis=-1)
                y = np.stack([batch[f] for f in self.features], axis=-1)


            elif mode =='factors' or mode == 'nearest':
                assert y_factors is not None, "y_factors cannot be 'None' if mode is 'factor'"
                # Build a y vector from the y_factors by interpolating the dataset statistics
                y_feature_space = DoomDataset().get_feature_sample(self.config.dataset_path, y_factors, features, 'std')
                if mode == 'nearest':
                    assert sample_from_dataset in ['train',
                                                   'validation'], "If mode is 'nearest' then 'sample_from_dataset' must be either 'train' or 'validation'"
                    # Use nearest neighbour
                    if self.nearest_features_tree is None:
                        loaded_features = DoomDataset().load_features(self.config.dataset_path, sample_from_dataset, self.features, self.output_size)
                        self.nearest_features_tree = cKDTree(loaded_features)

                    y = np.zeros_like(y_feature_space)
                    for s in range(self.config.batch_size):
                        distance, nearest_index = self.nearest_features_tree.query(y_feature_space[s])
                        y[s, :] = loaded_features[nearest_index]
                else:
                    # Don't use a dataset point
                    y = y_feature_space
            elif mode == 'direct':
                assert y_batch is not None, "y_batch cannot be 'None' if mode is 'direct'"
                # the y_batch is provided externally
                y = y_batch

        # Z Sampling
        if seed is not None:
            np.random.seed(seed)
        if freeze_z:
            if z_override is not None:
                z_sample = z_override
            else:
                z_sample = np.random.uniform(-1, 1, [1, self.config.z_dim]).astype(
                np.float32)
            z_batch = np.repeat(z_sample, self.config.batch_size, axis=0)
        else:
            if z_override is not None:
                z_batch = z_override
            else:
                z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                    np.float32)
        np.random.seed()
        # Sample Generation
        net_input = {self.z: z_batch}
        if self.use_features:
            net_input[self.y] = y
        result = self.session.run([self.G_rescaled],
                         feed_dict=net_input)[0]
        # Postprocessing and saving
        if postprocess:
            if save is False:
                return DataTransform.postprocess_output(result, self.maps, folder=None)
            elif save == 'PNG':
                if folder is None:
                    # Use default folder
                    return DataTransform.postprocess_output(result, self.maps, true_samples=x_true_batch,
                                                            feature_vector=y)
                else:
                    return DataTransform.postprocess_output(result, self.maps, true_samples=x_true_batch, feature_vector=y, folder=folder)
            elif save == 'WAD':
                return DataTransform.build_levels(result, self.maps, self.config.batch_size)
        return result

    def evaluate_samples_distribution(self, input_subset=None, n=1, sample_from_dataset='train', postprocess=True):
        """
        Generates n maps for each feature vector in the dataset and computes the feature vector.
        If input_subset is None, then the levels are taken from the dataset specified by sample_from_dataset, otherwise
        input_subset should be a dictionary of type {'key':{'feature1':float, 'feature2':float, ...}} containing
        the levels that have to be used as input.
        In that case, the returned value is a tuple containing a list of level names and the list of the corresponding generated levels

        Returns (as Numpy Arrays):
         * A list of level names (since the dataset gets shuffled when loaded). Size [levels]
         * The list of input features for the true levels. Size [levels, features(input)]
         * The list of the other features for the true levels that are not used as network input. Size [levels, features(~input)]
         * The list of generated features, for each level and for each n sample. Size [levels, features(input), N]
         * The list of generated features, for each level and for each n sample, that are not used as network input. Size [levels, features(~input), N]
         * The list of noise vectors used in each sampling, for each level and for each sample. Size [levels, z_dim, N]

        The index of the first dimension of each returned vector matches the index of the others,
        so level_names[4] is the name of the level having features results_true[4] that are used to generate results_gen[4] using noise vector result_noise[4], etc.
        The same is valid for the last dimension N.
        :param n: how many samples (different noise vector z) to generate for each true level.
        :param sample_from_dataset: 'train' or 'validation' set
        :param postprocess: [True] if true, rescales and denoise data to match the inputs (for example, clamps the values of the floormap to either 0 or 255)
        :return: if input_subset is None: level names, true features input, additional true features, generated features, additional generated features, noise vectors. Otherwise level names, generated features (all)
        """
        results_true = list()
        results_true_oth = list()
        results_gen = list()
        results_gen_oth = list()
        results_z = list()
        level_names = list()



        # Feature columns definition
        oth_features = [f for f in all_features.features_for_evaluation if f not in self.features]

        import WAD_Parser.WADFeatureExtractor
        feature_extractor = WAD_Parser.WADFeatureExtractor.WADFeatureExtractor()
        self.initialize_and_restore()

        if input_subset is None:
            # Loop through the dataset
            train_set_iter, valid_set_iter = self.load_dataset()
            train_set_size, validation_set_size = DoomDataset().get_dataset_count(self.config.dataset_path)

            chosen_iter = train_set_iter if sample_from_dataset == 'train' else valid_set_iter
            chosen_size = train_set_size if sample_from_dataset == 'train' else validation_set_size
            self.session.run([chosen_iter.initializer])
            next_batch = chosen_iter.get_next()

            batch_counter = 0
            errors = 0
            while True:
                try:
                    batch = self.session.run(next_batch)

                    y_batch_true = np.stack([batch[f] for f in self.features], axis=-1) if self.use_features else []
                    y_batch_true_oth = np.stack([batch[f] for f in oth_features], axis=-1)
                    names_batch = np.array(batch['path_json'])


                    record = {'true_batch': y_batch_true, 'names_batch': names_batch, 'gen_batch': list(), 'z_batch': list(), 'true_names': list(), 'gen_batch_oth': list(), 'true_batch_oth': y_batch_true_oth}
                    for z_sampling in range(n):
                        print("Batch {} of {}, sample {} of {}".format(batch_counter+1, chosen_size//self.config.batch_size + 1, z_sampling+1, n))
                        # Generate a new batch of maps
                        z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                            np.float32)
                        x_batch_gen = self.sample(mode='direct', y_batch=y_batch_true, z_override=z_batch, postprocess=postprocess).astype(np.uint8)
                        y_batch_gen = list()  # Batches of generated features (inputs to the networks)
                        y_batch_gen_oth = list()  # Batches of generated features (other features, not in input)
                        for l, level in enumerate(x_batch_gen):
                            # Compute level features
                            try:
                                # Compute all features from the generated level maps
                                y_gen_all = feature_extractor.extract_features_from_maps(
                                            floormap=level[:, :, self.maps.index('floormap')],
                                            wallmap=level[:, :, self.maps.index('wallmap')],
                                            thingsmap=level[:, :, self.maps.index('thingsmap')])
                                # Select only the features that are inputs to the network
                                y_gen = np.asarray([y_gen_all[name] for name in self.features])
                                # Other features
                                y_gen_oth = np.asarray([y_gen_all[name] for name in oth_features])
                            except:
                                # Fill the feature record with NaNs indicating a failure in calculation
                                y_gen = np.full_like(self.features, np.nan)
                                y_gen_oth = np.full_like(oth_features, np.nan)
                                errors+=1
                            y_batch_gen.append(y_gen)
                            y_batch_gen_oth.append(y_gen_oth)
                        record['gen_batch'].append(np.asarray(y_batch_gen))
                        record['gen_batch_oth'].append(np.asarray(y_batch_gen_oth))
                        record['z_batch'].append(np.asarray(z_batch))
                    # Now unrolling the batches of features into lists of levels
                    true_batch = record['true_batch']
                    true_batch_oth = record['true_batch_oth']
                    generated_batch = np.asarray(record['gen_batch']).transpose((1,2,0))
                    generated_batch_oth = np.asarray(record['gen_batch_oth']).transpose((1,2,0))
                    noise_batch = np.asarray(record['z_batch']).transpose((1,2,0))
                    if self.use_features:
                        for names, true, true_oth, gen, gen_oth, noise in zip(names_batch, true_batch, true_batch_oth, generated_batch, generated_batch_oth, noise_batch):
                            level_names.append(names)
                            results_true.append(true)
                            results_true_oth.append(true_oth)
                            results_gen.append(gen)
                            results_gen_oth.append(gen_oth)
                            results_z.append(noise)
                    else:
                        # True batch and Gen batch are empty, since network don't use any features. All the features are in
                        for names, true_oth, gen_oth, noise in zip(names_batch, true_batch_oth, generated_batch_oth, noise_batch):
                            level_names.append(names)
                            results_true_oth.append(true_oth)
                            results_gen_oth.append(gen_oth)
                            results_z.append(noise)
                    batch_counter += 1

                except tf.errors.OutOfRangeError:
                    print("Done")
                    if errors>0:
                        print("{} levels have NaN feature data since feature extraction didn't work on some samples.".format(errors))
                    break
            return np.asarray(level_names), np.asarray(results_true), np.asarray(results_true_oth), np.asarray(results_gen), np.asarray(results_gen_oth), np.asarray(results_z)
        else:
            results = dict()
            for lname, level in input_subset.items():
                print("Generating {} levels...".format(n))
                # Preparing the result record
                results[lname] = dict()
                for f in all_features.features_for_evaluation:
                    if f not in results[lname]:
                        results[lname][f] = list()
                n_batches = int(np.ceil(n/self.config.batch_size))
                for i in range(n_batches):
                    y_batch_true = np.repeat([level[f] for f in self.features], self.config.batch_size,
                                             axis=-1).transpose((1, 0)) if self.use_features else []
                    z_batch = np.random.uniform(-1, 1, [self.config.batch_size, self.config.z_dim]).astype(
                        np.float32)
                    x_batch_gen = self.sample(mode='direct', y_batch=y_batch_true, z_override=z_batch,
                                              postprocess=postprocess).astype(np.uint8)
                    for l_id in range(min(n, self.config.batch_size)):
                        genlevel=x_batch_gen[l_id,...]
                        # Compute level features
                        try:
                            # Compute all features from the generated level maps
                            y_gen_all = feature_extractor.extract_features_from_maps(
                                floormap=genlevel[:, :, self.maps.index('floormap')],
                                wallmap=genlevel[:, :, self.maps.index('wallmap')],
                                thingsmap=genlevel[:, :, self.maps.index('thingsmap')])
                            # Adding generated features to the results

                        except:
                            # Fill the feature record with NaNs indicating a failure in calculation
                            y_gen_all = np.full_like(all_features.features_for_evaluation, np.nan)

                        for f in y_gen_all:
                            results[lname][f].append(y_gen_all[f])

            names = list()
            generated = list()
            for r in results:
                names.append(r)
                generated.append(results[r])
            return np.asarray(names), np.asarray(generated)


    def get_samples_by_name(self, names):
        """ get a dictionary of names (path_json field) and associates a true sample for each value"""
        self.load_dataset()
        # Loop through the true dataset
        train_set_iter, valid_set_iter = self.load_dataset()
        self.session.run([train_set_iter.initializer])
        next_batch = train_set_iter.get_next()
        while True:
            try:
                batch = self.session.run(next_batch)
                for name in names:
                    b_name = name if isinstance(name, bytes) else bytes(name, 'utf-8')
                    if b_name in batch['path_json']:
                        names[name] = {key: batch[key][batch['path_json']==b_name] for key in batch}
            except tf.errors.OutOfRangeError:
                break
        return names

    def compute_quality_metrics(self, s_gen, s_true):
        """
        Compute several metrics between a batch of generated sample and true sample corresponding on the same y vector.
        :param s_gen: the batch of generated samples
        :param s_true: the batch of true samples
        :return:
        """
        s_gen = s_gen.astype(np.uint8)
        from skimage.measure import compare_ssim
        corner_error = lambda x, y: np.power((np.power(x - y, 2) / (x * y)), 0.5)

        metrics = {}
        entropy_diff = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        ssim = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        enc_error = np.zeros(shape=(self.config.batch_size, len(self.maps), 1))
        floor_corner_error = np.zeros(shape=(self.config.batch_size, 1))
        walls_corner_error = np.zeros(shape=(self.config.batch_size, 1))

        for s in range(self.config.batch_size):
            # TODO: Add topological features here if makes any sense
            metrics_true = quality_metrics(s_true[s,:,:,:], self.maps)
            metrics_gen = quality_metrics(s_gen[s,:,:,:], self.maps)

            for m, mname in enumerate(self.maps):
                ssim[s, m, :] = compare_ssim(s_gen[s, :, :, m], s_true[s, :, :, m])
                entropy_diff[s, m, :] = metrics_gen['entropy_{}'.format(mname)] - metrics_true['entropy_{}'.format(mname)]
                # Encoding error should be > 0 only for generated samples
                enc_error[s, m, :] = metrics_gen['encoding_error_{}'.format(mname)]
                if mname == 'floormap':
                    floor_corner_error[s,:] = corner_error(metrics_true['corners_{}'.format(mname)], metrics_gen['corners_{}'.format(mname)])
                if mname == 'wallmap':
                    walls_corner_error[s,:] = corner_error(metrics_true['corners_{}'.format(mname)], metrics_gen['corners_{}'.format(mname)])

                metrics["entropy_mae_{}".format(mname)] = np.mean(entropy_diff[:,m,:])
                metrics["similarity_{}".format(mname)] = np.mean(ssim[:,m,:])
                metrics["encoding_error_{}".format(mname)] = np.mean(enc_error[:,m,:])
        metrics["entropy_mae"] = np.mean(entropy_diff)
        metrics["similarity"] = np.mean(ssim)
        metrics["encoding_error"] = np.mean(enc_error)
        metrics["floor_corner_error"] = np.mean(floor_corner_error)
        metrics["walls_corner_error"] = np.mean(walls_corner_error)
        return metrics


def define_flags():
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 10000, "Epoch to train [10000]")
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
    flags.DEFINE_string("dataset_path", "./dataset/128-one-floor.meta", "Path to the .meta file of the corresponding .TFRecord datasets")
    flags.DEFINE_integer("height", 128, "Target sample height")
    flags.DEFINE_integer("width", 128, "Target sample width")
    flags.DEFINE_integer("z_dim", 100, "Dimension for the noise vector in input to G [100]")
    flags.DEFINE_integer("batch_size", 32, "Batch size")
    flags.DEFINE_integer("save_net_every", 100, "Number of training iterations after which the next is saved")
    flags.DEFINE_integer("seed", None, "Seed for generating samples. If None random noise is applied to the generator input [None]")
    flags.DEFINE_boolean("train", False, "enable training if true")
    flags.DEFINE_boolean("generate", False, "If true, generate some levels with a fixed y value and seed and save them into ./generated_samples")
    flags.DEFINE_boolean("use_sigmoid", True,
                         "If true, uses sigmoid activations for G outputs, if false uses Tanh. Data will be normalized accordingly")
    flags.DEFINE_boolean("use_wgan", True, "Whether to use the Wesserstein GAN model or the standard GAN")
    flags.DEFINE_boolean("use_gradient_penalty", True, "Whether to use the gradient penalty from WGAN_GP architecture or the WGAN weight clipping")
    flags.DEFINE_string("checkpoint_dir", "./artifacts/checkpoint/", "Directory name to save the checkpoints [./artifacts/checkpoint/]")
    flags.DEFINE_string("summary_folder", "./artifacts/tensorboard_results/",
                        "Directory name to save the temporary files for visualization [./artifacts/tensorboard_results/]")
    flags.DEFINE_string("generated_folder", "./artifacts/generated_samples/",
                        "Directory name to save the generated samples [./artifacts/generated_samples/]")
    flags.DEFINE_string("ref_sample_folder", "./artifacts/",
                        "Directory name to save the generated samples [./artifacts/]")
    return flags.FLAGS



FLAGS = define_flags()
with tf.Session() as s:
    # Layers and inputs are defined in network_architecture.py
    features = architecture.features
    maps = architecture.maps
    g_layers = architecture.g_layers
    d_layers = architecture.d_layers
    print("Building the network...")
    gan = DoomGAN(session=s,
                  config=FLAGS, features=features, maps=maps,
                  g_layers = g_layers, d_layers = d_layers
                  )
    show_all_variables()

    if __name__ == '__main__':
        if FLAGS.train:
            print("Starting training process...")
            gan.train(FLAGS)
        else:
            feat_factors = [-1 for f in features]
            if FLAGS.generate:
                factors = np.random.normal(0.5, 0.1, size=(FLAGS.batch_size, len(features)))
                gan.initialize_and_restore()
                gan.sample(mode='dataset', y_factors=factors, postprocess=True, save='PNG', seed=FLAGS.seed, folder=FLAGS.generated_folder)


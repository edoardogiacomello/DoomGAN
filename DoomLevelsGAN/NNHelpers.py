import tensorflow as tf
import math
import itertools

def show_all_variables():
  model_vars = tf.trainable_variables()
  tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def leaky_relu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def elu(x):
    return tf.nn.elu(x)



def PS(X, r, color=False):
    """
    Code from https://github.com/tetrachrome/subpixel
    :param X:
    :param r:
    :param color:
    :return:
    """
    def _phase_shift(I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  #
        bsize, a * r, b * r
        return tf.reshape(X, (bsize, a * r, b * r, 1))

    # Main OP that you can arbitrarily use in you tensorflow code
    if color:
      Xc = tf.split(3, 3, X)
      X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
      X = _phase_shift(X, r)
    return X


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02,
           name="conv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if with_w:
            return conv, w
        else:
            return conv


def conv2d_transposed(input_, output_shape,
                      k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02,
                      name="deconv2d", with_w=False, remove_artifacts=False):
    with tf.variable_scope(name):
        if not remove_artifacts:
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            try:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                strides=[1, stride_h, stride_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                        strides=[1, stride_h, stride_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if with_w:
                return deconv, w
            else:
                return deconv
        else:
            # We do the transpose convolution by separating the upsampling from the convolution iself.
            # Since the convolution output size is computed ad out = ceil(in/stride), then for obtaining the desired size
            # we have to upscale to size = floor(H*stride)
            upscale_size = [math.floor(output_shape[1]*stride_h), math.floor(output_shape[2]*stride_w)]
            upscaled = tf.image.resize_images(input_, size=upscale_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            w = tf.get_variable('w', [k_h, k_w, upscaled.get_shape()[-1], output_shape[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(upscaled, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            if with_w:
                return conv, w
            else:
                return conv


def linear_layer(x, output_size, scope=None,stddev=0.02, bias_start=0.0, with_w=False, on_cpu_memory=False):
    if on_cpu_memory:
        with tf.device('/cpu:0'):
            return linear_layer(x,output_size,scope,stddev,bias_start,with_w,on_cpu_memory=False)

    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(x, W) + b, W
    else:
        return tf.matmul(x, W) + b

def concatenate_features(l, y):
    y_new_shape = [y.get_shape()[0].value, 1, 1, y.get_shape()[-1].value]
    ones_shape = [l.get_shape()[0].value, l.get_shape()[1].value, l.get_shape()[2].value, y.get_shape()[-1].value]
    y_ =  tf.reshape(y, y_new_shape) * tf.ones(ones_shape, dtype=tf.float32)
    return tf.concat([l,y_], axis=-1)


def visualize_activations(name, layers, input_number):
    """
    Shows the activations for a given layer
    :param name:
    :param samples: a list of layer outputs, each of them having shape (batch, height, width, depth)
    :param input_number:
    :return: a summary for the layer activation and a summary for the chosen input
    """
    # Select the given samples from the batch
    summaries = []
    for l_id, layer in enumerate(layers):
        layer = tf.transpose(tf.slice(layer, begin=[input_number, 0, 0, 0], size=[1, -1, -1, -1]), [3, 1, 2, 0])
        summaries += [visualize_samples(name+'{}'.format(l_id), layer)]
    return summaries

def visualize_samples(name, samples):
    """
    Helper for visualizing the samples in a grid that is sqrt(batch)*sqrt(batch).
    """
    batch = samples.get_shape()[0].value
    channels = samples.get_shape()[-1].value

    unstacked_samples = tf.unstack(samples, axis=0)

    # Finding the size of the image grid (in number of images)
    rows = math.floor(math.sqrt(batch))
    cols = math.ceil(batch/rows)

    grid = [unstacked_samples[r*cols:(r+1)*cols] for r in range(rows)]

    #If the last grid is not complete, it may raise an error when evaluated.

    while len(grid[-1]) < cols:
        grid[-1].append(tf.zeros_like(unstacked_samples[0]))

    tensor= tf.expand_dims(tf.concat([tf.concat(r,axis=1) for r in grid], axis=0), axis=0)
    # If the input sample has 2 or more than 3 channels, display each channel separately
    tensor = tf.squeeze(tf.stack([tf.expand_dims(channel, axis=-1) for channel in tf.unstack(tensor, axis=-1)]), axis=1)
    return tf.summary.image(name, tensor, max_outputs=channels)


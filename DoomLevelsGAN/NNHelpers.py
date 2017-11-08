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
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_h, stride_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv2d_transposed(input_, output_shape,
                      k_h=5, k_w=5, stride_h=2, stride_w=2, stddev=0.02,
                      name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, stride_h, stride_w, 1], data_format='NHWC')

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, stride_h, stride_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

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
        return tf.matmul(x, W) + b, W, b
    else:
        return tf.matmul(x, W) + b

def visualize_activations(name, input, max_outputs=None, tiled=False):
    """Helper for visualizing the activations of a convolutional layer in form of an image. If tiled is false then
    the number of filters can be any and they will be shown in a squared image where each filter is displayed next to the other.
    If tiled is True the blocks that makes the image are formed from each single activation for the same portion of image
    """
    max_outputs = input.get_shape()[0].value if max_outputs is None else max_outputs
    unstacked_filters = tf.unstack(input, axis=3)
    number_of_blocks = math.ceil(math.sqrt(len(unstacked_filters)))
    grid = list()
    if tiled is False:
        # We tile every single filter adjacently, we need to find a square to inscribe them
        for row in [unstacked_filters[i:i + number_of_blocks] for i in range(0, len(unstacked_filters), number_of_blocks)]:
            for filter in row:
                shape = filter.get_shape()
                pass
            grid.append(tf.concat(row, axis=2))
        input = tf.expand_dims(tf.concat(grid, axis=1), axis=-1)
    else:
        input = tf.depth_to_space(input, number_of_blocks)
    return tf.summary.image(name, input, max_outputs)

def visualize_samples(name, input):
    """
    Helper for visualizing the samples in a grid that is sqrt(batch)*sqrt(batch).
    If the batch size is not evenly divisible by an integer number, than each sample is visualized separately
    """
    batch = input.get_shape()[0].value
    unstacked_samples = tf.unstack(input, axis=0)
    number_of_blocks = math.ceil(math.sqrt(len(unstacked_samples)))

    # Finding the size of the image grid (in number of images)
    rows = math.floor(math.sqrt(batch))
    cols = math.ceil(batch/rows)

    grid = [unstacked_samples[r*cols:(r+1)*cols] for r in range(rows)]

    #If the last grid is not complete, it may raise an error when evaluated.

    while len(grid[-1]) < cols:
        grid[-1].append(tf.zeros_like(unstacked_samples[0]))

    tensor= tf.expand_dims(tf.concat([tf.concat(r,axis=1) for r in grid], axis=0), axis=0)

    return tf.summary.image(name, tensor, max_outputs=1)

def reverse_enumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    """
    return itertools.izip(reversed(xrange(len(iterable))), reversed(iterable))
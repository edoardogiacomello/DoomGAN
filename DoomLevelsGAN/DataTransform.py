import tensorflow as tf
import DoomDataset as dd
import numpy as np
import skimage.io

def scaling_maps(x, map_names, dataset_path, use_sigmoid=True):
    """
    Compute the scaling of every map based on their .meta statistics (max and min)
     to bring all values inside (0,1) or (-1,1)
    :param x: the input vector. shape(batch, width, height, len(map_names))
    :param map_names: the name of the feature maps for .meta file lookup
    :param use_sigmoid: if True data will be in range 0,1, if False it will be in -1;1
    :return: a normalized x vector
    """
    a = 0 if use_sigmoid else -1
    b = 1
    meta = dd.DoomDataset().read_meta(dataset_path)
    max = tf.constant([meta['maps'][m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(x.get_shape())
    min = tf.constant([meta['maps'][m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(x.get_shape())
    return a + ((x-min)*(b-a))/(max-min)



def scaling_features(y, feature_names, dataset_path, use_sigmoid=True):
    """
    Compute the scaling of every feature based on their .meta statistics (max and min)
     to bring all values inside (0,1) or (-1,1)
    :param y: the input vector. shape(batch, len(feature_names))
    :param feature_names: the name of the features for .meta file lookup
    :param use_sigmoid: if True data will be in range 0,1, if False it will be in -1;1
    :return: a normalized y vector
    """
    a = 0 if use_sigmoid else -1
    b = 1
    meta = dd.DoomDataset().read_meta(dataset_path)
    max = tf.constant([meta['features'][f]['max'] for f in feature_names], dtype=tf.float32) * tf.ones(y.get_shape())
    min = tf.constant([meta['features'][f]['min'] for f in feature_names], dtype=tf.float32) * tf.ones(y.get_shape())
    return a + ((y-min)*(b-a))/(max-min)

def scaling_maps_inverse(g, map_names, dataset_path, used_sigmoid=True):
    """
    Compute the inverse transformation of 'scaling_maps' function:
    Take as input a list of feature maps in range (0,1) or (-1,1) and rescales them back to the original range based on
    the .meta file of the dataset given as input.
    :param g: The input tensor, typically the generator network output
    :param map_names: List of map names
    :param dataset_path: Path of the dataset
    :param used_sigmoid: if True data is considered to be in range 0,1, in -1;1 if False
    :return: a Tensor of rescaled values
    """
    a = 0 if used_sigmoid else -1
    b = 1
    meta = dd.DoomDataset().read_meta(dataset_path)
    max = tf.constant([meta['maps'][m]['max'] for m in map_names], dtype=tf.float32) * tf.ones(g.get_shape())
    min = tf.constant([meta['maps'][m]['min'] for m in map_names], dtype=tf.float32) * tf.ones(g.get_shape())
    return min + ((g-a)*(max-min))/(b-a)

def postprocess_output(g, maps, folder = './generated_samples/'):
    """
    This function takes care of denoising network output and other steps, in order to prepare the maps to be processed
    by the WADEditor
    :return:
    """
    # floormap is an enumeration, so low levels may be barely visible. It could be better to rescale based on the
    # min/max value of the map instead of the whole dataset.
    for s_id, sample in enumerate(g):
        for m, mapname in enumerate(maps):
            feature_map = sample[:,:,m]
            feature_map = np.around(feature_map)
            if mapname == 'floormap':
                feature_map = (feature_map > 0)*255
            # Saving
            skimage.io.imsave(folder + 'level{}_map_{}.png'.format(s_id, mapname), feature_map.astype(np.uint))

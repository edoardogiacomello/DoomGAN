import tensorflow as tf
import DoomDataset as dd


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

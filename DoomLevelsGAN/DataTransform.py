import tensorflow as tf
import DoomDataset as dd
import numpy as np
import skimage.io
from WAD_Parser.WADEditor import WADWriter, WADReader
import os

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

def postprocess_output(g, maps, folder = './artifacts/generated_samples/', true_samples = None, feature_vector = None):
    """
    This function takes care of denoising network output and other steps, in order to prepare the maps to be processed
    by the WADEditor. If "folder" is set to none, output is not saved but returned in a ndarray instead
    :param g: Batch of generated maps to be rescaled or postprocessed, size (batch, width, height, m)
    :param maps: list of map names of len(maps) = m
    :param folder: String or None. If not none, then the samples are saved in that folder
    :param true_samples: (Optional) The true samples to be saved along the generated one for visual comparison
    :param feature_vector: (Optional) The feature vector that generated the samples to be saved in a .txt file
    :return: The processed batch
    """
    processed_output = g.copy()
    for s_id, sample in enumerate(g):
        for m, mapname in enumerate(maps):
            feature_map = sample[:,:,m]
            if mapname == 'heightmap':
                feature_map = ((feature_map-feature_map.min())*255.0)/(feature_map.max()-feature_map.min())
            if mapname == 'floormap':
                feature_map = (feature_map >= 255/2)*255
            if mapname == 'wallmap':
                feature_map = (feature_map >= 255/2)*255
            feature_map = np.around(feature_map)
            processed_output[s_id, :, :, m] = feature_map

            # Saving
            if folder is not None:
                os.makedirs(folder, exist_ok=True)
                skimage.io.imsave(folder + 'sample{}_map_{}_generated.png'.format(s_id, mapname), processed_output[s_id, :, :, m].astype(np.uint))
                if true_samples is not None:
                    skimage.io.imsave(folder + 'sample{}_map_{}_true.png'.format(s_id, mapname), true_samples[s_id, :, :, m].astype(np.uint))
        if (folder is not None) and (feature_vector is not None):
            np.savetxt(folder + "sample{}_y_features.txt".format(s_id), feature_vector[s_id])
    return processed_output





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

def get_interpolated_y_batch(start, end, batch_size):
    """
    Return a sferical interpolated sample of size (batch, features) for which
     result[0] is "start"
     result[batch_size] is "end"
     middle values are feature space vectors interpolated at step 1/batch_size
    :param start: the starting feature vector
    :param end: the ending feature vector
    :param batch_size: first dimension of returned vector, also determines the interpolation step
    :return: ndarray of size (batch_size, start.shape()[-1] == end.shape()[-1])
    """
    result = np.zeros(shape=(batch_size, start.shape[-1]))
    for s, step in enumerate(np.linspace(0, 1, num=batch_size)):
        result[s] = slerp(step, start, end)
    return result


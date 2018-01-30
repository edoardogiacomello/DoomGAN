import tensorflow as tf
import DoomDataset as dd
import numpy as np
import skimage.io
from WAD_Parser.WADEditor import WADWriter, WADReader

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
    by the WADEditor. If "folder" is set to none, output is not saved but returned in a ndarray instead
    :param g:
    :param maps:
    :param folder:
    :return:
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

            # Saving
            processed_output[s_id,:,:,m] = feature_map
            if folder is not None:
                skimage.io.imsave(folder + 'level{}_map_{}.png'.format(s_id, mapname), feature_map.astype(np.uint))
    return processed_output

def build_levels(rescaled_g, maps, batch_size, tmp_folder = '/tmp/doomgan/', call_node_builder = True, level_images_path='./generated_samples/'):
    """
    Post-processes a rescaled network output and saves the corresponding wad file.
    :param rescaled_g: Rescaled network output, in the same scale of the dataset
    :param batch_size:
    :param tmp_folder: temp folder where to store the wad file
    :return: The path of the newly created wad file
    """
    # Create a new WAD
    writer = WADWriter()
    heightmap = None
    wallmap = None
    thingsmap = None
    floormap = None
    rescaled_g = postprocess_output(rescaled_g, maps, folder=None)
    for index in range(batch_size):
        for m, map in enumerate(maps):
            heightmap = rescaled_g[index,:,:,m] if map == 'heightmap' else heightmap
            wallmap = rescaled_g[index,:,:,m] if map == 'wallmap' else wallmap
            thingsmap = rescaled_g[index,:,:,m] if map == 'thingsmap' else thingsmap
            floormap = rescaled_g[index,:,:,m] if map == 'floormap' else floormap
        writer.add_level(name='MAP{i:02d}'.format(i=index + 1))
        writer.from_images(heightmap, floormap=floormap, wallmap=wallmap, thingsmap=thingsmap, save_debug=level_images_path, level_coord_scale=32)
    import os
    os.makedirs(tmp_folder, exist_ok=True)
    wad_path = tmp_folder+'generated_wad.wad'
    writer.save(wad_path, call_node_builder=call_node_builder)
    return wad_path

def extract_features_from_net_output(rescaled_g, features, maps, batch_size, tmp_folder = '/tmp/doomgan/', call_node_builder = False):
    """
    This function creates a .WAD from the net output (still not playable unless "call_node_builder" is set to true) and returns a vector
    containing the features extracted from the newly created WAD file.
    :param rescaled_g:
    :param features:
    :param maps:
    :param batch_size:
    :param tmp_folder:
    :param call_node_builder:
    :return:
    """
    extracted_features = np.zeros(shape=(batch_size, len(features)))
    reader = WADReader()
    wad_path = build_levels(rescaled_g, maps, batch_size, tmp_folder, call_node_builder)
    wad = reader.extract(wad_path)
    for l, level in enumerate(wad['levels']):
        for f, feature in enumerate(features):
            extracted_features[l,f] = level['features'][feature]
    return extracted_features
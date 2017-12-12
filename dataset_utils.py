import networkx as nx
from collections import namedtuple
from collections import defaultdict
import os
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from scipy import misc
import warnings
import skimage.io
import glob
import WAD_Parser.Dictionaries.Features as Features
from scipy import stats




###### This part contains all the variables concerning the dataset #####
# This is the tile dictionary, containing also the pixel colors for image conversion
channel_grey_interval = (255 // 16)
channel_s_interval = (255//2)
channel_g_interval = (255//13)
# channel s contains: empty, wall, floor, (stairs encoded as floor)
# channel g contains: enemy, weapon, ammo, health, barrel, key, start, teleport, decorative, teleport, exit

from WAD_Parser.WADEditor import WADReader

# TODO: This script is a bit messy, refactor it together with meta_utils


tile_tuple = namedtuple("tile_tuple", ["pixel_color", "tags"])
tiles_greyscale = {
        "-": tile_tuple(( 0), ["empty", "out of bounds"]),  # is black
        "X": tile_tuple((1 * channel_grey_interval), ["solid", "wall"]),  # maroon
        ".": tile_tuple((2 * channel_grey_interval), ["floor", "walkable"]), # coral
        ",": tile_tuple((3 * channel_grey_interval), ["floor", "walkable", "stairs"]), # Beige
        "E": tile_tuple((4 * channel_grey_interval), ["enemy", "walkable"]), # red
        "W": tile_tuple((5 * channel_grey_interval), ["weapon", "walkable"]), # Blue
        "A": tile_tuple((6 * channel_grey_interval), ["ammo", "walkable"]), # Cyan
        "H": tile_tuple((7 * channel_grey_interval), ["health", "armor", "walkable"]), # Green
        "B": tile_tuple((8 * channel_grey_interval), ["explosive barrel", "walkable"]), # Magenta
        "K": tile_tuple((9 * channel_grey_interval), ["key", "walkable"]), # Teal
        "<": tile_tuple((10 * channel_grey_interval), ["start", "walkable"]), # Lavender
        "T": tile_tuple((11 * channel_grey_interval), ["teleport", "walkable", "destination"]), # Olive
        ":": tile_tuple((12 * channel_grey_interval), ["decorative", "walkable"]), # Grey
        "L": tile_tuple((13 * channel_grey_interval), ["door", "locked"]), # Brown
        "t": tile_tuple((14 * channel_grey_interval), ["teleport", "source", "activatable"]), # Mint
        "+": tile_tuple((15 * channel_grey_interval), ["door", "walkable", "activatable"]), # Orange
        ">": tile_tuple((16 * channel_grey_interval), ["exit", "activatable"]) # White
    }

tiles = {
        "-": tile_tuple((0,   0,   0), ["empty", "out of bounds"]),  # is black
        "X": tile_tuple((128, 0,   0), ["solid", "wall"]),  # maroon
        ".": tile_tuple((255, 215, 180), ["floor", "walkable"]), # coral
        ",": tile_tuple((255, 250, 200)	, ["floor", "walkable", "stairs"]), # Beige
        "E": tile_tuple((230, 25,  75), ["enemy", "walkable"]), # red
        "W": tile_tuple((0,   130, 200), ["weapon", "walkable"]), # Blue
        "A": tile_tuple((70,  240, 240)	, ["ammo", "walkable"]), # Cyan
        "H": tile_tuple((60,  180, 75), ["health", "armor", "walkable"]), # Green
        "B": tile_tuple((240, 50,  230), ["explosive barrel", "walkable"]), # Magenta
        "K": tile_tuple((0,   128, 128), ["key", "walkable"]), # Teal
        "<": tile_tuple((230, 190, 255)	, ["start", "walkable"]), # Lavender
        "T": tile_tuple((128, 128, 0), ["teleport", "walkable", "destination"]), # Olive
        ":": tile_tuple((128, 128, 128)	, ["decorative", "walkable"]), # Grey
        "L": tile_tuple((170, 110, 40)	, ["door", "locked"]), # Brown
        "t": tile_tuple((170, 255, 195)	, ["teleport", "source", "activatable"]), # Mint
        "+": tile_tuple((245, 130, 48)	, ["door", "walkable", "activatable"]), # Orange
        ">": tile_tuple((255, 255, 255)	, ["exit", "activatable"]) # White
    }

grey_to_rgb = [
[0,   0,   0], 
[128, 0,   0], 
[255, 215, 180],
[255, 250, 200],
[230, 25,  75],
[0,   130, 200],
[70,  240, 240],
[60,  180, 75],
[240, 50,  230],
[0,   128, 128],
[230, 190, 255],
[128, 128, 0], 
[128, 128, 128],
[170, 110, 40],	
[170, 255, 195],
[245, 130, 48],	
[255, 255, 255]]

gray_to_sg = [
[[0*channel_s_interval],[0 *channel_g_interval]], #empty
[[1*channel_s_interval],[0 *channel_g_interval]], #wall
[[2*channel_s_interval],[0 *channel_g_interval]], #floor
[[2*channel_s_interval],[0 *channel_g_interval]], #stairs
[[2*channel_s_interval],[1 *channel_g_interval]], #enemy
[[2*channel_s_interval],[2 *channel_g_interval]], #weapon
[[2*channel_s_interval],[3 *channel_g_interval]], #ammo
[[2*channel_s_interval],[4 *channel_g_interval]], #health
[[2*channel_s_interval],[5 *channel_g_interval]], #barrel
[[2*channel_s_interval],[6 *channel_g_interval]], #key
[[2*channel_s_interval],[7 *channel_g_interval]], #start
[[2*channel_s_interval],[8 *channel_g_interval]], #teleport dest
[[2*channel_s_interval],[9 *channel_g_interval]], #decorative
[[2*channel_s_interval],[10*channel_g_interval]], #door_locked
[[2*channel_s_interval],[11*channel_g_interval]], #teleport_src
[[2*channel_s_interval],[12*channel_g_interval]], #door_unlocked
[[2*channel_s_interval],[13*channel_g_interval]], #exit
]



def tf_from_greyscale_to_sg(images):
    """
    Converts a batch of grayscale images in range [0,255] to a dual channel representation where:
    channel s contains: empty, wall, floor, (stairs encoded as floor) [0,255]
    channel g contains: enemy, weapon, ammo, health, barrel, key, start, teleport, decorative, teleport, exit [0,255]
    :param images:
    :return:
    """
    images = tf.divide(images, tf.constant(255.0, dtype=tf.float32))
    # FIXME: Probably this is not working anymore
    images = tf.to_int32(tf_from_grayscale_to_tilespace(images))
    palette = tf.constant(gray_to_sg, tf.float32)
    images = tf.squeeze(images, axis=-1)
    sg_images = tf.gather(palette, images)
    return tf.squeeze(sg_images, axis=-1)


def tf_from_grayscale_to_tilespace(images, channels=1):
    """
    Converts a batch of inputs from the floating point representation [0,1] to the tilespace representation [0,1,..,n_tiles]
    :param images:
    :return:
    """
    # Rescale the input to [0,255]
    rescaled = images * tf.constant(255.0, dtype=tf.float32)
    channel_slice = []
    for c in range(channels):
        if channels == 1:
            chosen_interval = channel_grey_interval
        if channels == 2:
            chosen_interval = channel_s_interval if c == 0 else channel_g_interval
        interval = tf.constant(chosen_interval, dtype=tf.float32)
        half_interval = tf.constant(chosen_interval / 2, dtype=tf.float32)
        # Here the image contains pixel values that does not correspond to anything in the encoding
        mod = tf.mod(rescaled, interval)  # this is the error calculated from the previous right value.
        # I.e. if the encoding is [0, 15, 30] and the generated value is [14.0, 16.0, 30.0] this is [14, 1, 0]
        div = tf.floor_div(rescaled, interval)
        # this is the encoded value if the error is less then half the sampling interval, right_value - 1 otherwise
        # E.g. (continuing the example) [0, 1, 2] while the correct encoding should be [1, 1, 2]
        mask = tf.floor(tf.divide(mod, half_interval))
        #  This mask tells which pixels are already right (0) or have to be incremented (1)
        # E.g [1, 0, 0]
        encoded = div + mask  # Since the mask can be either 0 or 1 for each pixel, the true encoding
        # will be obtained by summing the two
        channel_slice.append(encoded)
    return tf.squeeze(tf.stack(channel_slice, axis=-1), axis=-2)

def tf_from_grayscale_to_rgb(tile_indices):
    """
    Converts a greyscale sample (encoded by tiles_greyscale) to a rgb image (encoded by tiles dict) for better
    visualization
    :param tile_indices: a Tensor with shape [batch, height, width, 1]
    :return: a Tensor with shape [batch, height, width, 3]
    """
    palette = tf.constant(grey_to_rgb, dtype=tf.float32)
    return tf.squeeze(tf.gather(palette, tf.to_int32(tile_indices)), axis=3)

def tf_from_grayscale_to_multichannel(images):
    """
    Converts a batch of images from a greyscale (0,1) representation to a space [batch, height, width, n_tiles] where
    each channel represent a type of tile, and the data can be either 0 or 1.
    :param tile_indices:
    :return:
    """
    encoded = tf.to_int32(tf_from_grayscale_to_tilespace(images))
    return tf.squeeze(tf.one_hot(encoded, depth=len(tiles_greyscale)), axis=3)

def tf_match_encoding(gen_output):
    """
    Correct the encoding of a noisy sample generated by the network
    :param gen_output:
    :return: An image batch of float in (0,1) which assume only n_tile values
    """
    g_encoded = tf_from_grayscale_to_tilespace(gen_output)
    # g_encoded contains the sample in label space (0 to n_tiles),
    # it has to be converted to be consistent with the true samples
    # Scaling to [0,1]
    return tf.multiply(g_encoded, tf.constant(255 // 16, dtype=tf.float32)) / tf.constant(255, dtype=tf.float32)

def tf_encode_feature_vectors(y, feature_names, dataset_path):
    """
    Reads metadata from the dataset .meta file and returns the normalized Tensor for the feature vector
    :param y: The unnomalized feature vector
    :param feature_names: The list of feature names, in same order as Dimension[-1] of y
    :param dataset_path: the path to the .TFRecord file (metadata are supposed to be at .TFRecord.meta )
    :return: The y tensor normalized depthwise
    """
    meta_path = dataset_path+'.meta'
    with open(meta_path, 'r') as meta_in:
        meta = json.load(meta_in)

    norm_channels = []
    for f_id, f_name in enumerate(feature_names):
        y_slice = tf.slice(y, begin=[0,f_id], size=[-1,1])
        feat_max = tf.constant(meta['features'][f_name]['max'], dtype=tf.float32)
        feat_min = tf.constant(meta['features'][f_name]['min'], dtype=tf.float32)
        norm_channels.append( (y_slice-feat_min)/(feat_max-feat_min))
    return tf.squeeze(tf.stack(norm_channels, axis=1))






    

class DatasetManager(object):


    def __init__(self, target_size=(512,512), min_size=(16,16)):
        """
        Class for reading/writing the dataset in .json/png <-> .TFRecords
        :param target_size: Final (for conversion) or expected (for TFRecord loading) sample size in PNG/tile format.
        """
        self.target_size = target_size
        self.meta = dict()
        self.min_size = min_size


    def _sample_to_TFRecord(self, json_record):
        # converting the record to a default_dict since it may does not contain some keys for empty values.
        json_record = defaultdict(lambda: "", json_record)

        feature_dict = dict()
        # Dinamically build a tf.train.Feature dictionary based on dataset Features
        for f in Features.features:
            if Features.features[f] == 'int':
                feat_type = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record[f])]))
            if Features.features[f] == 'float':
                feat_type = tf.train.Feature(float_list=tf.train.FloatList(value=[float(json_record[f])]))
            if Features.features[f] == 'string':
                feat_type = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record[f])]))
            feature_dict[f] = feat_type

        # Doing the same for the maps
        for m in Features.map_paths:
            feature_dict[Features.map_paths[m]] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[json_record[Features.map_paths[m]].tobytes()]))

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def _TFRecord_to_sample(self, TFRecord):
        feature_dict = dict()
        # Dinamically build a tf.train.Feature dictionary based on dataset Features
        for f in Features.features:
            if Features.features[f] == 'int':
                feat_type = tf.FixedLenFeature([],tf.int64)
            if Features.features[f] == 'float':
                feat_type = tf.FixedLenFeature([],tf.float32)
            if Features.features[f] == 'string':
                feat_type = tf.FixedLenFeature([],tf.string)
            feature_dict[f] = feat_type

        # Doing the same for the maps
        for m in Features.map_paths:
            feature_dict[Features.map_paths[m]] = tf.FixedLenFeature([],tf.string)

        parsed_features = tf.parse_single_example(TFRecord, feature_dict)

        # Decoding the maps
        for m in Features.map_paths:
            parsed_img = tf.decode_raw(parsed_features[Features.map_paths[m]], tf.uint8)
            parsed_img = tf.reshape(parsed_img, shape=(self.target_size[0], self.target_size[1]))
            parsed_features[Features.map_paths[m]] = parsed_img
        return parsed_features

    def _update_meta(self, level):
        """
        Updates the metadata for a level.
        Metadata contain information such the global number of entries, the min and max values for each feature, etc.
        :param level: The json record for a level
        :return:
        """
        def _compute_feature(level, feature, type):
            if type == 'int64' or type == 'float' or type == 'int':
                if feature not in self.meta['features']:
                    self.meta['features'][feature] = dict()
                feat_dict = self.meta['features'][feature]
                feat_dict['type'] = type
                feat_dict['min'] = float(level[feature]) if 'min' not in feat_dict else min(feat_dict['min'], float(level[feature]))
                feat_dict['max'] = float(level[feature]) if 'max' not in feat_dict else max(feat_dict['max'], float(level[feature]))
                feat_dict['avg'] = float(level[feature]) if 'avg' not in feat_dict else feat_dict['avg'] + (float(level[feature]) - feat_dict['avg'])/float(self.meta['count'])

        def _compute_map(level, map):
                if map not in self.meta['maps']:
                    self.meta['maps'][map] = dict()
                feat_dict = self.meta['maps'][map]
                feat_dict['type'] = str(level[map].dtype)
                feat_dict['min'] = float(level[map].min()) if 'min' not in feat_dict else min(feat_dict['min'],
                                                                                            float(level[map].min()))
                feat_dict['max'] = float(level[map].max()) if 'max' not in feat_dict else max(feat_dict['max'],
                                                                                              float(level[map].max()))
                feat_dict['avg'] = float(level[map].mean()) if 'avg' not in feat_dict else feat_dict['avg'] + (float(
                    level[map].mean()) - feat_dict['avg']) / float(self.meta['count'])




        self.meta['features'] = dict() if 'features' not in self.meta else self.meta['features']
        self.meta['maps'] = dict() if 'maps' not in self.meta else self.meta['maps']
        self.meta['count'] = 1 if 'count' not in self.meta else self.meta['count'] +1
        import WAD_Parser.Dictionaries.Features as Features
        for f in Features.features:
            _compute_feature(level, f, Features.features[f])
        for m in Features.map_paths:
            _compute_map(level, Features.map_paths[m])

    def _pad_image(self, image):
        """Center pads an image, adding a black border up to "target size" """
        assert image.shape[0] <= self.target_size[0], "The image to pad is bigger than the target size"
        assert image.shape[1] <= self.target_size[1], "The image to pad is bigger than the target size"
        padded = np.zeros((self.target_size[0],self.target_size[1]), dtype=np.uint8)
        offset = (self.target_size[0] - image.shape[0])//2, (self.target_size[1] - image.shape[1])//2  # Top, Left
        padded[offset[0]:offset[0]+image.shape[0], offset[1]:offset[1]+image.shape[1]] = image
        return padded

    def load_TFRecords_database(self, path):
        """Returns a tensorflow dataset from the .tfrecord file specified in path"""
        dataset = tf.contrib.data.TFRecordDataset(path)
        dataset = dataset.map(self._TFRecord_to_sample, num_threads=9)
        return dataset


    def convert_to_TFRecords(self, record_list, dataset_root, output_path):
        """
        Pack the whole image dataset into the TFRecord standardized format and saves it at the specified output path.
        Pads each sample to the target size, DISCARDING the samples that are larger.
        Also save meta information in a separated file.
        :return: None.
        """
        print("{} levels loaded.".format(len(record_list)))


        with tf.python_io.TFRecordWriter(output_path) as writer:
            saved_levels = 0
            for counter, level in enumerate(record_list):
                # Reading the maps
                for path in Features.map_paths:
                    map_img = skimage.io.imread(dataset_root + level[path], mode='L')
                    padded = self._pad_image(map_img)
                    level[Features.map_paths[path]] = padded
                self._update_meta(level)
                sample = self._sample_to_TFRecord(level)
                writer.write(sample.SerializeToString())
                saved_levels += 1

                if counter % (len(record_list) // 100) == 0:
                    print("{}% completed.".format(round(counter / len(record_list) * 100)))
            print("{} levels Saved.".format(saved_levels))

        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as meta_out:
            # Saving metadata
            json.dump(self.meta, meta_out)
            print("Metadata saved to {}".format(meta_path))

    def remove_outliers(self, samples):
        """
        Removes some outliers based on data observation
        :param points:
        :return:
        """

        def outlier(p):
            sec_area_too_big = p['sector_area_avg'] > 0.5e8
            sectors_too_stretch = p['sector_aspect_ratio_avg'] > 20
            too_many_lines_per_sector = p['lines_per_sector_avg'] > 60
            wrong_walkable_percentage = p['walkable_percentage'] < 0 or p['walkable_percentage'] > 1
            too_thin_levels = p['aspect_ratio'] > 5
            too_many_sectors = p['number_of_sectors'] > 550
            too_many_floors = p['floors'] > 40
            return sec_area_too_big or sectors_too_stretch or too_many_lines_per_sector or wrong_walkable_percentage \
                   or too_thin_levels or too_many_sectors or too_many_floors

        clean = [s for s in samples if not outlier(s)]
        print("Removed {} outliers".format(len(samples) - len(clean)))
        return clean

    def filter_dataset(self, json_dataset):
        with open(json_dataset, 'r') as jin:
            data = json.load(jin)
        # Removing some outliers
        clean = self.remove_outliers(data)
        # filtering based on the size: each pixel is converted in 32 doom map units.
        data = [d for d in clean if d['width'] <= self.target_size[0] * 32 and d['height'] <= self.target_size[1] * 32
                and d['width'] >= self.min_size[0] * 32 and d['height'] >= self.min_size[1] * 32]
        print("Removed {} oversized samples".format(len(clean) - len(data)))
        return data



def plot_dataset_stats(data, features):
    points = np.array([[d[f] for f in features] for d in data])
    import pandas as pd
    import seaborn as sb
    pd_dataset = pd.DataFrame(points, columns=features)
    g = sb.pairplot(pd_dataset)
    import matplotlib.pyplot as plt
    plt.show()

def build_dataset():
    dm = DatasetManager(target_size=(128, 128))
    dm.filter_dataset('/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json')
    clean_dataset = dm.filter_dataset('/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json')
    # plot_dataset_stats(clean_dataset, features=['number_of_sectors', 'aspect_ratio', 'sector_area_avg', 'lines_per_sector_avg', 'sector_aspect_ratio_avg', 'floors', 'nonempty_percentage', 'walkable_percentage'])
    dm.convert_to_TFRecords(clean_dataset, '/run/media/edoardo/BACKUP/Datasets/DoomDataset/', '/run/media/edoardo/BACKUP/Datasets/DoomDataset/128newmeta.TFRecords')
    # rebuild_database('/run/media/edoardo/BACKUP/Datasets/DoomDataset/Processed/','/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json')

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

# In order to add a new feature modify the following parts:
    # meta description in datasetmanager.__init__
    # _update_meta  in datasetmanager
    # TFRecord to Sample in datasetmanager
    # sample to TFRecord in datasetmanager


###### This part contains all the variables concerning the dataset #####
# This is the tile dictionary, containing also the pixel colors for image conversion
channel_grey_interval = (255 // 16)
channel_s_interval = (255//2)
channel_g_interval = (255//13)
# channel s contains: empty, wall, floor, (stairs encoded as floor)
# channel g contains: enemy, weapon, ammo, health, barrel, key, start, teleport, decorative, teleport, exit

from WAD_Parser.WADEditor import WADReader



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


class MetaReader(object):
    def __init__(self, tfrecord_path):
        self.path = tfrecord_path+'.meta'
        with open(self.path, 'r') as meta_in:
            self.meta = json.load(meta_in)
    def count(self):
        return int(self.meta['count'])

    def label_average(self, feature_names, y_like_array):
        norm_channels = []
        for f_id, f_name in enumerate(feature_names):
            y_slice = tf.slice(y_like_array, begin=[0, f_id], size=[-1, 1])
            feat_max = tf.constant(self.meta['features'][f_name]['max'], dtype=tf.float32, shape=y_slice.get_shape())
            feat_min = tf.constant(self.meta['features'][f_name]['min'], dtype=tf.float32, shape=y_slice.get_shape())
            feat_avg = tf.constant(self.meta['features'][f_name]['avg'], dtype=tf.float32, shape=y_slice.get_shape())
            norm_channels.append((feat_avg - feat_min) / (feat_max - feat_min))
        return tf.squeeze(tf.stack(norm_channels, axis=1))

    

class DatasetManager(object):


    def __init__(self, path_to_WADs_folder='./WADs', relative_to_json_files=[], target_size=(512,512), target_channels=1):
        """
        Utility class for extracting metadata from the tile/grid representation, representation conversion (e.g to PNG)
        and TFRecord conversion an loading.

        :param path_to_WADs_folder: path to the /WADs/ folder when converting a dataset
        :param relative_to_json_files: list of paths to the json databases from the wad folder. eg: ['doom/Doom.json']
        :param target_size: Final (for conversion) or expected (for TFRecord loading) sample size in PNG/tile format.
        """
        self.G = nx.DiGraph()
        self.json_files = relative_to_json_files
        self.root = path_to_WADs_folder
        self.target_size = target_size
        self.target_channels = target_channels
        self.meta = dict()
        self.feature_sets = dict() # This dict keeps track of every different value for the string features, so it's possible to count unique values

    def _get_absolute_path(self, relative_path):
        """
        Given a relative path of type: "./WADs/....." returns the absolute path considering the root specified in init
        :param relative_path:
        :return:
        """
        # since tile_path begins with './WADs/'" we prepend our root
        if relative_path.startswith('./WADs/'):
            return relative_path.replace('./WADs/', self.root)
        else:
            return relative_path.replace('./', self.root)

    def extract_size(self):
        """For every level listed into the json files, adds the 'height' and 'width' information"""
        for j in self.json_files:
            with open(self.root + j, 'r') as jin:
                levels = json.load(jin)
            for level in levels:
                tile_path = self._get_absolute_path(level['tile_path'])
                # Open the tile file
                with open(tile_path) as level_file:
                    lines = [line.strip() for line in level_file.readlines()]
                    level['height'] = len(lines)
                    level['width'] = len(lines[0])
            with open(self.root+j+'.updt', 'w') as jout:
                json.dump(levels, jout)

    def _grid_to_matrix(self):
        pass

    def _grid_to_image(self, path):
        """
        Convert (renders) a tile representation of a level to a PNG
        :param path: Path of the .txt file
        :return: Pillow Image object
        """
        with open(path, 'r') as tilefile:
            lines = [line.strip() for line in tilefile.readlines()]
        n_lines = len(lines)
        n_cols = len(lines[0])
        mode = 'L' if self.target_channels == 1 else 'RGB'
        tile_set = tiles_greyscale if self.target_channels == 1 else tiles

        img = Image.new(mode, (n_lines, n_cols), "black")  # create a new black image

        ip = 0  # pixel counters
        jp = 0
        for i in lines:  # for every tile:
            for j in i:
                c = tile_set[j].pixel_color
                img.putpixel((ip, jp), c)
                jp += 1
            jp = 0
            ip += 1
        return img

    def convert_to_images(self):
        """
        For every level listed in the json databases, creates an image render from the grid/tile representation of the level
        and saves it into the /<gamename>/Processed - Rendered/ folder.
        It also updates the json database with img_path, width and height columns
        :return: None
        """
        for j in self.json_files:
            with open(self.root + j, 'r') as jin:
                levels = json.load(jin)
            for level in levels:
                tile_path = self._get_absolute_path(level['tile_path'])
                # We change to the Image subfolder
                img_path = tile_path.replace('Processed', 'Processed - Rendered').replace('.txt','.png')
                img_folder = '/'.join(img_path.split('/')[:-1])+'/'

                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                img = self._grid_to_image(tile_path)
                img.save(img_path, 'PNG')
                # Update the json file with the new path
                # [img_path contains the new root so it may be not consistent with the other paths already in db]
                level['img_path'] = level['tile_path'].replace('Processed', 'Processed - Rendered').replace('.txt','.png')
                # Also update the level size
                level['width'] = img.size[0]
                level['height'] = img.size[1]
            with open(self.root + j, 'w') as jout:
                json.dump(levels, jout)

    def _sample_to_TFRecord(self, json_record, image):
        # converting the record to a default_dict since it may does not contain some keys for empty values.
        json_record = defaultdict(lambda: "", json_record)
        features = {
            'author': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['author'])])),
            'description': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['description'])])),
            'credits': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['credits'])])),
            'base': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['base'])])),
            'editor_used': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['editor_used'])])),
            'bugs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['bugs'])])),
            'build_time': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['build_time'])])),
            'creation_date': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['creation_date'])])),
            'file_url': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['file_url'])])),
            'game': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['game'])])),
            'category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['category'])])),
            'title': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['title'])])),
            'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['name'])])),
            'path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['path'])])),
            'svg_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['svg_path'])])),
            'tile_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['tile_path'])])),
            'img_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(json_record['img_path'])])),
            'page_visits': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record['page_visits'])])),
            'downloads': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record['downloads'])])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record['height'])])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record['width'])])),
            'rating_count': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(json_record['rating_count'])])),
            'rating_value': tf.train.Feature(float_list=tf.train.FloatList(value=[float(json_record['rating_value'])])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        }

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _TFRecord_to_sample(self, TFRecord):
        features = {
            'author': tf.FixedLenFeature([],tf.string),
            'description': tf.FixedLenFeature([],tf.string),
            'credits': tf.FixedLenFeature([],tf.string),
            'base': tf.FixedLenFeature([],tf.string),
            'editor_used': tf.FixedLenFeature([],tf.string),
            'bugs': tf.FixedLenFeature([],tf.string),
            'build_time': tf.FixedLenFeature([],tf.string),
            'creation_date': tf.FixedLenFeature([],tf.string),
            'file_url': tf.FixedLenFeature([],tf.string),
            'game': tf.FixedLenFeature([],tf.string),
            'category': tf.FixedLenFeature([],tf.string),
            'title': tf.FixedLenFeature([],tf.string),
            'name': tf.FixedLenFeature([],tf.string),
            'path': tf.FixedLenFeature([],tf.string),
            'svg_path': tf.FixedLenFeature([],tf.string),
            'tile_path': tf.FixedLenFeature([],tf.string),
            'img_path': tf.FixedLenFeature([],tf.string),
            'page_visits': tf.FixedLenFeature([],tf.int64),
            'downloads': tf.FixedLenFeature([],tf.int64),
            'height': tf.FixedLenFeature([],tf.int64),
            'width': tf.FixedLenFeature([],tf.int64),
            'rating_count': tf.FixedLenFeature([],tf.int64),
            'rating_value': tf.FixedLenFeature([],tf.float32),
            'image': tf.FixedLenFeature([],tf.string)
        }

        parsed_features = tf.parse_single_example(TFRecord, features)
        parsed_img = tf.decode_raw(parsed_features['image'], tf.uint8)
        parsed_img = tf.reshape(parsed_img, shape=(self.target_size[0], self.target_size[1], self.target_channels) )
        parsed_features['image'] = parsed_img
        return parsed_features

    def _update_meta(self, level):
        """
        Updates the metadata for a level.
        Metadata contain information such the global number of entries, the min and max values for each feature, etc.
        :param level: The json record for a level
        :return:
        """
        def _compute_feature(level, feature, type):
            if feature not in self.meta['features']:
                self.meta['features'][feature] = dict()
            feat_dict = self.meta['features'][feature]
            feat_dict['type'] = type
            if type == 'int64' or type=='float':
                feat_dict['min'] = float(level[feature]) if 'min' not in feat_dict else min(feat_dict['min'], float(level[feature]))
                feat_dict['max'] = float(level[feature]) if 'max' not in feat_dict else max(feat_dict['max'], float(level[feature]))
                feat_dict['avg'] = float(level[feature]) if 'avg' not in feat_dict else feat_dict['avg'] + (float(level[feature]) - feat_dict['avg'])/float(self.meta['count'])
            if type == 'string':
                if feature not in self.feature_sets:
                    self.feature_sets[feature] = dict()
                entry_count = self.feature_sets[feature]
                if (feature not in level):
                    return
                # Increment the count for the current value of the feature
                entry_count[level[feature]] = 1 if level[feature] not in entry_count else entry_count[level[feature]] + 1
                feat_dict['count'] = len(entry_count)





        self.meta['features'] = dict() if 'features' not in self.meta else self.meta['features']
        self.meta['count'] = 1 if 'count' not in self.meta else self.meta['count'] +1
        _compute_feature(level, 'page_visits', 'int64')
        _compute_feature(level, 'downloads', 'int64')
        _compute_feature(level, 'height', 'int64')
        _compute_feature(level, 'width', 'int64')
        _compute_feature(level, 'rating_count', 'int64')
        _compute_feature(level, 'rating_value', 'int64')
        _compute_feature(level, 'author', 'string')


    def _pad_image(self, image):
        """Center pads an image, adding a black border up to "target size" """
        assert image.shape[0] <= self.target_size[0], "The image to pad is bigger than the target size"
        assert image.shape[1] <= self.target_size[1], "The image to pad is bigger than the target size"
        padded = np.zeros((self.target_size[0],self.target_size[1],self.target_channels), dtype=np.uint8)
        offset = (self.target_size[0] - image.shape[0])//2, (self.target_size[1] - image.shape[1])//2  # Top, Left
        padded[offset[0]:offset[0]+image.shape[0], offset[1]:offset[1]+image.shape[1],:] = image
        return padded


    def convert_to_TFRecords(self, record_list, output_path, max_size, min_size=(16, 16)):
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
                too_big = int(level['width']) > max_size[0] or int(level['height']) > max_size[1]
                too_small = int(level['width']) < min_size[0] or int(level['height']) < min_size[1]
                if too_big or too_small:
                    continue
                # TODO: Continue
                image = skimage.io.imread(self._get_absolute_path(level['img_path']), mode='L')
                if self.target_channels == 1:
                    image = np.expand_dims(image,-1)
                padded = self._pad_image(image)
                self._update_meta(level)
                sample = self._sample_to_TFRecord(level, padded)
                writer.write(sample.SerializeToString())
                saved_levels+=1

                if counter % (len(record_list)//100) == 0:
                    print("{}% completed.".format(round(counter/len(record_list)*100)))
            print("Levels saved: {}, out of range: {}".format(saved_levels, len(record_list)-saved_levels))

        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as meta_out:
            # TODO: This can be done when analyzing the levels
            # Embed author encoding in the metadata
            self.meta['encoding'] = dict()
            self.meta['encoding']['authors'] = {name: id for id, name in enumerate(self.feature_sets['author'])}
            # Saving metadata
            json.dump(self.meta, meta_out)
            print("Metadata saved to {}".format(meta_path))

    def load_TFRecords_database(self, path):
        """Returns a tensorflow dataset from the .tfrecord file specified in path"""
        dataset = tf.contrib.data.TFRecordDataset(path)
        dataset = dataset.map(self._TFRecord_to_sample, num_threads=9)
        return dataset



def recompute_dataset_features(db_root, relative_to_json_dbs, out_dataset_folder, from_id = 0):
    """ Recompute dataset features given an already available level dataset without having to extract each zip file again """
    dataset = list()
    updated_dataset = list()
    for jf in relative_to_json_dbs:
        json_path = db_root+jf
        with open(json_path, 'r') as file:
            dataset += json.load(file)

    for i, level in enumerate(dataset):
        if i < from_id:
            # Skip sample
            continue
        # FIXME: Remove this when the dataset is completed
        # fixing paths
        level['path'] = level['path'].replace('./WADs/Doom/', '')
        level['path'] = level['path'].replace('./WADs/DoomII/', '')
        wad_full_path = db_root +level['path']
        # Fix: Remove old unused features
        del level['width']
        del level['height']
        del level['img_path']
        del level['svg_path']
        del level['tile_path']

        try:
            print("{}/{}".format(i, len(dataset)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wad = WADReader().extract(wad_full_path, save_to=out_dataset_folder, root_path=db_root, update_record=level)

            for level in wad['levels']:
                # Now we have a complete level record directly in the features dictionary. We can just fetch it for faster indexing
                updated_dataset.append(level['features'])


        except:
            print('Error parsing {}'.format(wad_full_path))

        if i % 10 == 0:
            # Saving partial result
            with open(db_root+'updated_dataset.json', 'w') as jout:
                json.dump(updated_dataset, jout)


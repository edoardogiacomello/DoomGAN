import json
import os
import csv

import WAD_Parser.Dictionaries.Features as Features
import numpy as np
import skimage.io as io
import tensorflow as tf
from collections import defaultdict
import scipy.stats as stats
from sklearn.model_selection import train_test_split



class DoomDataset():
    """
    Utility class for loading and managing the doom dataset in .json/png representation.
    The dataset is structured as follow:
    - a root <dataset_root>/
    - a .json database <dataset_root>/dataset.json in which all the level features are stored
    - a <dataset_root>/Original/ folder, containing the .WAD files for each level
    - a <dataset_root>/Processed/ folder, containing:
        -<zip_name>_<wad_name>_<slot_name>.json file containing the features for a level (one row dataset.json)
        -<zip_name>_<wad_name>_<slot_name>_<feature_map>.png image(s) containing the feature map for a level

    These files are indexed from the dataset.json starting from <dataset_root>.
    E.g. a path could be "Processed/myzip_MyLevel_E1M1_floormap.png"
    The feature maps dimensions are (width/32, height/32), since 32 is the diameter of the smallest thing that exists on Doom.
    Each pixel value is an uint8, dicrectly encoding a value (ie. the "thing type index" for thingsmap; 1,2,3,4.. for
    the "floormap" enumeration or the floor_height value for the heightmap.

    Dataset can also be stored in a .TFRecord file (and this is the format DoomGAN uses to read the dataset);
    this is useful if you want to previously filter a dataset perhaps selecting only <128x128 levels and padding smaller ones.
    This way you pack all the dataset in a single .TFRecord file and its relative .meta file, containing aggregated data
    for each feature, such as min/max/avg value along the samples that have been selected in order to further normalize the data.
    """
    def __init__(self):
        self.root = None  # Root when reading from json
        self.json_db = None

    def read_from_json(self, json_db):
        """
        Reads the db from it's .json database and returns a list of level records.
        :param json_db: the .json file record in <dataset_root>/<yourdb>.json
        :return:
        """
        assert os.path.isfile(json_db), "Json database not found at {}".format(json_db)
        self.json_db = json_db
        self.root = '/'.join(json_db.split('/')[0:-1]) + '/'
        assert os.path.isdir(self.root+'Processed/'), '"Processed" directory not found in {}'.format(self.root)
        assert os.path.isdir(self.root+'Original/'), '"Processed" directory not found in {}'.format(self.root)
        levels = list()
        with open(json_db, 'r') as fin:
            levels += json.load(fin)
        return levels

    def read_from_TFRecords(self, tfrecords_path, target_size):
        """Returns a tensorflow dataset from the .tfrecord file specified in path"""
        dataset = tf.contrib.data.TFRecordDataset(tfrecords_path)
        dataset = dataset.map(lambda l: self._TFRecord_to_sample(l, target_size), num_threads=9)
        return dataset

    def get_path_of(self, feature_field):
        """
        Return the full path for a given feature, such a featuremap.
         Example: .get_path_of(level["wallmap"]) = /path/to/root/Processed/yourlevel_wallmap.png
        :param feature: The field containing the relative path of the featuremap/wad file you want to obtain
        :return: A file path
        """
        assert self.root is not None, "No root specified for this database. Are you sure you opened it with read_from_json()?"
        return self.root+feature_field

    def rebuild_database(self, root, database_name='database.json'):
        """
        Reads all the .json files inside the given <root>/"Processed" folder and rebuilds the database
        :param root: The root folder of the database. All the <level>.json and <level>.png must be stored into the "Processed" subfolder
        :param database_name: Filename of the resulting file. It will be saved to <root>/<database_name>
        :return:
        """
        import glob
        assert os.path.isdir(root + 'Processed/'), '"Processed" directory not found in {}'.format(self.root)
        self.root = root
        processed_folder = self.root+'Processed/'
        jsons = glob.glob(processed_folder + '*.json')
        database = list()
        for i, j in enumerate(jsons):
            with open(j, 'r') as jin:
                database += [json.load(jin)]
            if i % 100 == 0:
                print("{} of {} records rebuilt".format(i, len(jsons)))
        with open(self.root+database_name, 'w') as jout:
            json.dump(database, jout)

    def read_meta(self, meta_path):
        assert os.path.isfile(meta_path), \
            ".meta file database not found at {}.".format(meta_path)
        with open(meta_path, 'r') as meta_in:
            return json.load(meta_in)

    def get_dataset_count(self, dataset_path):
        """
        Returns the training and validation sample count
        :param dataset_path:
        :return:
        """
        return self.read_meta(dataset_path)['t_count'], self.read_meta(dataset_path)['v_count']

    def recompute_features(self, root, old_json_db, new_json_db):
        """
        This functions gets a json database and re-computes all the features and the maps for each WAD referenced by it,
        parsing all the WAD files from scratch. Useful when you are adding or editing features.
        WARNING: This function may overwrite your data, so make sure to keep a backup before executing it.
        :param root:
        :param old_json_db:
        :param new_json_db:
        :return:
        """
        import itertools
        import WAD_Parser.WADEditor as we
        import warnings
        old_records = self.read_from_json(old_json_db)
        print("Sorting levels..")
        sorted_input = sorted(old_records, key=lambda x: x['path'])
        # Grouping the old levels by .WAD path

        wad_records = itertools.groupby(sorted_input, key=lambda x: x['path'])
        new_records = list()
        for i, (wad, record) in enumerate(wad_records):
            # Assuming that wad-level features are the same for each level
            record = next(record)
            wad_reader = we.WADReader()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    parsed_wad = wad_reader.extract(wad_fp=root+record['path'], save_to=root+'Processed/', update_record=record,
                                                root_path=root)
                except Exception as e:
                    print("Error parsing {}: {}".format(root+record['path'], e))
            for level in parsed_wad['levels']:
                new_records.append(level['features'])
            if len(new_records) % (len(sorted_input)//100) == 0:
                print("{}% completed...".format(len(new_records)//(len(sorted_input)//100)))

        with open(new_json_db, 'w') as json_out:
            json.dump(new_records, json_out)
        print("Saved {} levels to {}".format(len(new_records), new_json_db))

    def to_csv(self, json_db_path, csv_path):
        """Converts a json dataset into csv representation, based on Features described in Features.py"""
        levels = self.read_from_json(json_db_path)
        with open(csv_path, 'w') as csvfile:
            keys = Features.features.keys()
            dict_writer = csv.DictWriter(csvfile, keys)
            dict_writer.writeheader()
            dict_writer.writerows(levels)
        print("Csv saved to: {}".format(csv_path))

    def filter_data(self, data, list_of_lambdas):
        for condition in list_of_lambdas:
            data = filter(condition, data)
        return list(data)


    def plot_joint_feature_distributions(self, path_or_data, features, constraints_lambdas=list(), cluster = False):
        """
        Plots the joint distribution for each couple of given feature
        :param path_or_data: (str or list) path of the json_db or the list of record containing data
        :param features: list of features to plot
        :return: None
        """
        import pandas as pd
        import seaborn as sb
        import matplotlib.pyplot as plt
        from sklearn import decomposition
        from sklearn import mixture
        data = self.read_from_json(path_or_data) if isinstance(path_or_data, str) else path_or_data
        data = self.filter_data(data, constraints_lambdas)
        points = np.array([[d[f] for f in features] for d in data])
        X=points

        # Print some stats
        by_col = np.transpose(X)
        for f, fname in enumerate(features):
            print("{}: \t mean={} \t std={} \t median={} \t min={} \t max={}".format(fname,by_col[f].mean(), by_col[f].std(), np.median(by_col[f]), by_col[f].min(), by_col[f].max()))

        if cluster:
            from sklearn.cluster import DBSCAN
            Y = DBSCAN(eps=0.3, min_samples=300).fit_predict(X)
            X = np.concatenate((X, np.expand_dims(Y, axis=-1)), axis=-1)
            # Plotting
            pd_dataset = pd.DataFrame(X, columns=features+['label'])
            g = sb.pairplot(pd_dataset, hue='label', plot_kws={"s": 10})
        else:
            pd_dataset = pd.DataFrame(X, columns=features)
            g = sb.pairplot(pd_dataset, plot_kws={"s": 10})
        return g

    def _add_maps_meta(self, meta, level):
        """
        Updates the dataset metadata given a new level, adding stats about its featuremaps.
        :param level: The json record for a level
        :return: the updated meta dictionary
        """

        import WAD_Parser.Dictionaries.Features as Features
        meta['maps'] = dict()
        for m in Features.map_paths:
            current_map = Features.map_paths[m]
            if current_map not in meta['maps']:
                meta['maps'][current_map] = dict()
            feat_dict = meta['maps'][current_map]
            feat_dict['type'] = str(level[current_map].dtype)
            feat_dict['min'] = float(level[current_map].min()) if 'min' not in feat_dict else min(feat_dict['min'],
                                                                                          float(level[current_map].min()))
            feat_dict['max'] = float(level[current_map].max()) if 'max' not in feat_dict else max(feat_dict['max'],
                                                                                          float(level[current_map].max()))
            feat_dict['avg'] = float(level[current_map].mean()) if 'avg' not in feat_dict else feat_dict['avg'] + (float(
                level[current_map].mean()) - feat_dict['avg']) / float(meta['count'])
        return meta

    def _feature_meta(self, level_records):
        """
        Compute the metadata for each feature of the given list of level records.
        Metadata contain descriptive statistical information about each feature. Image maps are not considered by this function.
        :param level_records:
        :return: dict()
        """
        meta = dict()
        meta['features'] = dict()
        meta['maps'] = dict()
        meta['count'] = len(level_records)

        import WAD_Parser.Dictionaries.Features as Features
        for f in Features.features:
            type = Features.features[f]
            if type in ['int64', 'float', 'int']:
                dtype = np.float64 if type == 'float' else np.int32
                values = np.asarray([v[f] for v in level_records], dtype=dtype)
                s = stats.describe(values)
                meta['features'][f] = dict()
                meta['features'][f]['min'] = s.minmax[0]
                meta['features'][f]['max'] = s.minmax[1]
                meta['features'][f]['mean'] = s.mean
                meta['features'][f]['var'] = s.variance
                meta['features'][f]['skewness'] = s.skewness
                meta['features'][f]['kurtosis'] = s.kurtosis
                meta['features'][f]['Q1'] = np.percentile(values, 25)
                meta['features'][f]['Q2'] = np.percentile(values, 50)
                meta['features'][f]['Q3'] = np.percentile(values, 75)
                # Saving every value as a scalar so it can be serialized by json
                for statname in meta['features'][f]:
                    meta['features'][f][statname] = np.asscalar(np.asarray(meta['features'][f][statname]))

        return meta

    def _pad_image(self, image, target_size):
        """Center pads an image, adding a black border up to "target size" """
        assert image.shape[0] <= target_size[0], "The image to pad is bigger than the target size"
        assert image.shape[1] <= target_size[1], "The image to pad is bigger than the target size"
        padded = np.zeros((target_size[0],target_size[1]), dtype=np.uint8)
        offset = (target_size[0] - image.shape[0])//2, (target_size[1] - image.shape[1])//2  # Top, Left
        padded[offset[0]:offset[0]+image.shape[0], offset[1]:offset[1]+image.shape[1]] = image
        return padded

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

    def _TFRecord_to_sample(self, TFRecord, target_size):
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
            parsed_img = tf.reshape(parsed_img, shape=(target_size[0], target_size[1]))
            parsed_features[Features.map_paths[m]] = parsed_img
        return parsed_features

    def to_TFRecords(self, json_db, output_path, validation_size, target_size=(128,128), constraints_lambdas=list()):
        """
        Pack the whole image dataset into the TFRecord standardized format and saves it at:
         <output_path>-train.TFRecord for the training set
         <output_path>-validation.TFRecord for the validation set
        Saves additional information about the data in a separated .meta file.
        Pads each sample to the target size, DISCARDING the samples that are larger.

        :param json_db: the .json database file to read features from
        :param output_path: path for the .TFRecords file
        :param validation_size: [0,1] relative portion of samples belonging to the validation set
        :param target_size: size (tuple) for the largest selected sample (smaller ones will be padded, largest are discarded)
        :param constraints_lambdas: list of lambdas defining constraint on data, EG. [lambda x: x['floors']==1]
        :return: None. Saves a .TFRecords file at the given output_path and a .meta json file at <output_path>.meta
        """
        # Reading the json dataset
        record_list = self.read_from_json(json_db)
        print("{} levels loaded.".format(len(record_list)))

        # Creating paths
        train_path = output_path+'-train.TFRecord'
        validation_path = output_path+'-validation.TFRecord'

        # Filtering data
        constraints_lambdas.append(lambda x: x["width"] <= 32*target_size[0])
        constraints_lambdas.append(lambda x: x["height"] <= 32*target_size[1])
        record_list = self.filter_data(record_list, constraints_lambdas)

        # Calculating meta for the scalar features
        meta = self._feature_meta(record_list)

        # Splitting into train and validation
        train_list, validation_list = train_test_split(record_list, test_size=validation_size)

        # Updating meta with train and validation count
        meta['t_count'] = len(train_list)
        meta['v_count'] = len(validation_list)

        with tf.python_io.TFRecordWriter(train_path) as train_writer:
            with tf.python_io.TFRecordWriter(validation_path) as validation_writer:
                saved_levels = 0
                for current_set in train_list, validation_list:
                    for level in current_set:
                        try:
                            # Reading the maps
                            for path in Features.map_paths:
                                map_img = io.imread(self.root + level[path], mode='L')
                                padded = self._pad_image(map_img, target_size=target_size)
                                level[Features.map_paths[path]] = padded
                            # Adding map meta to global meta (cannot load all the dataset in memory for computing stats)
                            meta = self._add_maps_meta(meta, level)
                            sample = self._sample_to_TFRecord(level)
                            if current_set is train_list:
                                train_writer.write(sample.SerializeToString())
                            else:
                                validation_writer.write(sample.SerializeToString())
                            saved_levels += 1
                        except:
                            print("Found an image that is larger than the target size. Skipping..")
                            continue
                        if saved_levels % (len(record_list) // 100) == 0:
                            print("{}% completed.".format(round(saved_levels / len(record_list) * 100)))
                print("{} levels saved.".format(saved_levels))
        meta_path = output_path + '.meta'
        with open(meta_path, 'w') as meta_out:
            # Saving metadata
            json.dump(meta, meta_out)
            print("Metadata saved to {}".format(meta_path))
        print("Done")

    def get_feature_sample(self, tf_dataset_path, factors, features, extremes='minmax'):
        """
        Returns a sample of a feature vector (y) given a list of feature names and an array of "factors" having the same shape of y (batch_size, len(features)).
        Each factor is a scalar relative to the corresponding batch sample and feature:
        If it's in [0,1] then the returned corresponding feature will range from the min value (factor = 0) to the max value (factor = 1)
        If the factor is -1 then the returned feature is the mean value for that feature.

        :param tf_dataset_path: The dataset to read data from
        :param factors: A vector of scalars in {-1, [0,1]}, having size (batch_size, len(features)).
        :param features: A list of feature names
        :param extremes: 'minmax' or 'std'
        :return: a vector y having the same shape of "factors".
        """
        assert factors.shape[-1] == len(features), "Length of factor and features array should be the same."
        meta = self.read_meta(tf_dataset_path)
        y = np.zeros_like(factors, dtype=np.float32)
        for f, f_name in enumerate(features):
            # Apply the feature mean where the factor is -1
            y[:,f] = np.where((factors[:,f] == -1), meta['features'][f_name]['mean'], y[:,f])

            # Apply the rescaled feature where the factor is between 0 and 1
            between = np.logical_and(factors[:,f] >= 0, factors[:,f] <= 1)
            a = 0
            b = 1
            f_min = meta['features'][f_name]['min'] if (extremes == 'minmax') else (meta['features'][f_name]['mean'] - np.sqrt(meta['features'][f_name]['var']))
            f_max = meta['features'][f_name]['max'] if (extremes == 'minmax') else (meta['features'][f_name]['mean'] + np.sqrt(meta['features'][f_name]['var']))

            y[:, f] = np.where(between, f_min + ((factors[:,f]-a)*(f_max-f_min))/(b-a), y[:, f])
        return y

    def get_feature_stats(self,tf_dataset_path, features, stat):
        """
        Returns an array containing the requested stat for the given set of features
        :param tf_dataset_path: path of the tfrecords dataset
        :param features: list of feature names
        :param stat: (str) name of the stat. can be 'min', 'max', 'mean', 'var', 'skewness', 'kurtosis'
        :return:
        """
        meta = self.read_meta(tf_dataset_path)
        stats = np.zeros(shape=(len(features)), dtype=np.float32)
        for f, fname in enumerate(features):
            stats[f] = meta['features'][fname][stat]
        return stats

    def get_dataset_path(self, meta_path, type):
        assert type in ['train', 'validation'], "Dataset type must be 'train' or 'validation'"
        dataset_path = meta_path.replace('.meta', '-{}.TFRecord'.format(type))
        assert os.path.isfile(dataset_path), "{} dataset not found at {}. Make sure you the file is accessible and it hasn't been renamed".format(type, dataset_path)
        return dataset_path



    def load_features(self, dataset_meta_path, dataset_type, feautre_names, sample_size):
        """
        Returns an array containing all the dataset rows corresponding to the given "feature_names".
        :param dataset_meta_path: path of the dataset .meta file
        :param dataset_type: Either 'train' or 'validation', needed for true dataset path lookup
        :param feautre_names: array containing the names of the requested features
        :param sample_size: dimension (in pixel) of the maps, needed for decoding the dataset file
        :return:
        """
        dataset_path = self.get_dataset_path(dataset_meta_path, dataset_type)
        train_count, validation_count = self.get_dataset_count(dataset_meta_path)
        count = train_count if dataset_type == 'train' else validation_count
        tf_dataset = self.read_from_TFRecords(dataset_path, target_size=sample_size)
        tf_dataset = tf_dataset.batch(count)

        iter = tf_dataset.make_one_shot_iterator()
        with tf.Session() as sess:
            data = sess.run([iter.get_next()])[0]
        return np.asarray([data[f] for f in feautre_names]).transpose()

    def generate_stats(self, dataset_path):
        dataset = DoomDataset()
        data = dataset.read_from_json(dataset_path)
        size_filter = lambda l: l['height']/32 <= 128 and l['width']/32 <= 128
        bound_things_number = lambda l: l['number_of_things'] < 1000 # Remove a few sample with a huge number of items
        bound_lines_per_sector = lambda l: l['lines_per_sector_avg'] < 150  # Removes a few samples with too many lines per sector
        bound_euler_number = lambda l: l['level_euler_number'] > -50  # Remove a few samples with too many holes

        data = dataset.filter_data(data, [size_filter,bound_things_number,bound_lines_per_sector,bound_euler_number])

        base_features = ['lines_per_sector_avg', 'number_of_things', 'walkable_area', 'walkable_percentage']
        level_features = [f for f in Features.features if f.startswith('level_') and not '_hu_' in f]
        floor_features = [f for f in Features.features if f.startswith('floors_') and (f.endswith('_mean'))]
        dataset.plot_joint_feature_distributions(data, features=base_features).savefig('./../dataset/statistics/128_base_features_no_outliers')
        dataset.plot_joint_feature_distributions(data, features=level_features).savefig('./../dataset/statistics/128_level_features_no_outliers')
        dataset.plot_joint_feature_distributions(data, features=floor_features).savefig('./../dataset/statistics/128_floor_features_no_outliers')

    def compare_features(self, data):
        import WAD_Parser.Dictionaries.Features as F
        import pandas as pd
        import seaborn as sns

        numerical_features = [f for f in F.features if f not in F.wad_features and F.features[f] != 'string']
        toplot = {k: data[k] for k in ["title"] + numerical_features}
        toplot["id"] = range(len(toplot["title"]))
        pdata = pd.DataFrame(toplot)
        sns.set(style="whitegrid")
        for feat in numerical_features:
            # Make the PairGrid
            g = sns.PairGrid(pdata.head(),
                             y_vars=feat, x_vars=["id"])

            # Draw a dot plot using the stripplot function
            g.map(sns.stripplot, palette="Reds_r", edgecolor="gray", orient="v")

            g.set(xlim=(-0.25, 6.5), xlabel="Sample")
            for ax in g.axes.flat:
                # Make the grid horizontal instead of vertical
                ax.xaxis.grid(False)
                ax.yaxis.grid(True)
            sns.despine(left=True, bottom=True)
            g.savefig('feature_comparison/feature_comparison_{}.png'.format(feat))
        print("show")

    def show_sample_batch(self, batch, channel, rows=6, cols=6, sample_size=128):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        for i in range(1, cols * rows + 1):
            img = batch[i - 1, :, :, channel]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img)
        plt.show()


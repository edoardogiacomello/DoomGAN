import json
import os

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

    def read_meta(self, tfrecord_path):
        meta_path = tfrecord_path + '.meta'
        assert os.path.isfile(meta_path), \
            ".meta file database not found at {}. No dataset statistics for normalizing the data".format(meta_path)
        with open(meta_path, 'r') as meta_in:
            return json.load(meta_in)

    def get_dataset_count(self, tfrecord_path):
        return self.read_meta(tfrecord_path)['count']

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
            del record['bounding_box_size']
            del record['nonempty_size']
            del record['nonempty_percentage']
            wad_reader = we.WADReader()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_wad = wad_reader.extract(wad_fp=root+record['path'], save_to=root+'Processed/', update_record=record,
                                            root_path=root)
            for level in parsed_wad['levels']:
                new_records.append(level['features'])
            if len(new_records) % (len(sorted_input)//100) == 0:
                print("{}% completed...".format(len(new_records)//(len(sorted_input)//100)))

        with open(new_json_db, 'w') as json_out:
            json.dump(new_records, json_out)
        print("Saved {} levels to {}".format(len(new_records), new_json_db))


if __name__ == '__main__':
    DoomDataset().recompute_features(root='/run/media/edoardo/BACKUP/Datasets/DoomDataset/',
                                     old_json_db='/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json',
                                     new_json_db='/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset_new.json'
                                     )



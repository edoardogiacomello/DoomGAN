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


import networkx as nx
from collections import namedtuple
from collections import defaultdict
import os
from PIL import Image
import json
import tensorflow as tf

# This is the tile dictionary, containing also the pixel colors for image conversion
tile_tuple = namedtuple("tile_tuple", ["pixel_color", "tags"])
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




class DatasetManager(object):
    """Extract metadata from the tile/grid representation of a level"""
    def __init__(self, path_to_WADs_folder, relative_to_json_files):
        self.G = nx.DiGraph()
        self.json_files = relative_to_json_files
        self.root = path_to_WADs_folder

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

        img = Image.new('RGB', (n_lines, n_cols), "black")  # create a new black image

        ip = 0  # pixel counters
        jp = 0
        for i in lines:  # for every tile:
            for j in i:
                c = tiles[j].pixel_color
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
            with open(self.root + j + '.updt', 'w') as jout:
                json.dump(levels, jout)

    def load(self, path):
        # TODO: Remove me or continue
        with open(path) as levelfile:
            for y, line in enumerate(levelfile):
                for x, tile in enumerate(line.strip()):
                    self.G.add_node((x,y), {"tile":tile})
                    if x > 0:
                        self.G.add_edge((x,y),(x-1,y), {'direction':'W'})
                        self.G.add_edge((x-1,y),(x,y), {'direction':'E'})
                    if y > 0:
                        self.G.add_edge((x,y),(x,y-1), {'direction':'N'})
                        self.G.add_edge((x,y-1),(x,y), {'direction':'S'})


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

    def _pad_image(self, image, target_size):
        """Center pads an image, adding a black border up to "target size" """
        assert image.size[0] <= target_size[0], "The image to pad is bigger than the target size"
        assert image.size[1] <= target_size[1], "The image to pad is bigger than the target size"
        padded = Image.new('RGB', target_size, "black")
        offset = (target_size[0] - image.size[0])//2, (target_size[1] - image.size[1])//2,
        padded.paste(image, offset)
        return padded



    def convert_to_TFRecords(self, output_path, target_size):
        """
        Pack the whole image dataset into the TFRecord standardized format and saves it at the specified output path.
        Pads each sample to the target size, DISCARDING the samples that are larger (this behaviour may change in future).
        Information about image size has to be stored separately.
        :return: None.
        """
        # Load the json files
        levels = []
        for j in self.json_files:
            with open(self.root + j, 'r') as jin:
                levels += json.load(jin)
        print("{} levels loaded.".format(len(levels)))
        with tf.python_io.TFRecordWriter(output_path) as writer:
            counter = 0
            for level in levels:
                counter += 1
                if int(level['width']) > target_size[0] or int(level['height']) > target_size[1]:
                    continue
                with Image.open(self._get_absolute_path(level['img_path'])) as image:
                    padded = self._pad_image(image, target_size)
                sample = self._sample_to_TFRecord(level, padded)
                writer.write(sample.SerializeToString())
                if counter % (len(levels)//100) == 0:
                    print("{}% completed.".format(round(counter/len(levels)*100)))




# DatasetManager('/run/media/edoardo/BACKUP/Datasets/DoomDataset/WADs/', ['Doom/Doom.json', 'DoomII/DoomII.json']).convert_to_TFRecords('/run/media/edoardo/BACKUP/Datasets/DoomDataset/lessthan512.TFRecords', target_size=(512,512))

# [scrapeUtils.json_to_csv(j) for j in ['/run/media/edoardo/BACKUP/Datasets/DoomDataset/WADs/Doom/Doom.json.updt', '/run/media/edoardo/BACKUP/Datasets/DoomDataset/WADs/DoomII/DoomII.json.updt']]

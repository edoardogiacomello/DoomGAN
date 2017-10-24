import networkx as nx
from collections import namedtuple
import os
from PIL import Image
import json

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




class MetaGenerator(object):
    """Extract metadata from the tile/grid representation of a level"""
    def __init__(self, path_to_WADs_folder, relative_to_json_files):
        self.G = nx.DiGraph()
        self.json_files = relative_to_json_files
        self.root = path_to_WADs_folder

    def extract_size(self):
        """For every level listed into the json files, adds the 'height' and 'width' information"""
        for j in self.json_files:
            with open(self.root + j, 'r') as jin:
                levels = json.load(jin)
            for level in levels:
                tile_path = level['tile_path']
                # since tile_path begins with './WADs/'" we prepend our root
                if tile_path.startswith('./WADs/'):
                    tile_path = tile_path.replace('./WADs/', self.root)
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
                tile_path = level['tile_path']
                # since tile_path begins with './WADs/'" we prepend our root
                if tile_path.startswith('./WADs/'):
                    tile_path = tile_path.replace('./WADs/', self.root)
                img_path = tile_path.replace('Processed', 'Processed - Rendered').replace('.txt','.png')

                # FIXME: this should create a directory but img_path is a file path
                # if not os.path.exists(img_path):
                #    os.makedirs(img_path)

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




# DatasetConverter('/run/media/edoardo/BACKUP/Datasets/DoomDataset/WADs/').convert_dataset_to_png()
MetaGenerator('/run/media/edoardo/BACKUP/Datasets/DoomDataset/WADs/',
              ['Doom/Doom.json',
               'DoomII/DoomII.json',
               ]).convert_to_images()
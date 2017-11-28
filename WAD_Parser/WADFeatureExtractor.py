import numpy as np
import skimage.draw as draw
from skimage import io
from scipy.ndimage.measurements import label


class WADFeatureExtractor(object):
    def __init__(self, level_dict):
        """
        Class for extracting a set of feature from a given WAD level
        :param level_dict: Wad in dictionary format as produced by WADReader.read()
        """
        self.level = level_dict


    def _rescale_coord(self, x_du, y_du):
        x_centered = x_du - self.level['features']['x_min']
        y_centered = y_du - self.level['features']['y_min']
        x = np.floor(x_centered / 32).astype(np.int32)
        y = np.floor(y_centered / 32).astype(np.int32)
        return x, y

    def _rescale_value(self, x, x_min, scale_factor):
        x_centered = x - x_min
        x = np.floor(x_centered / scale_factor).astype(np.int32)
        return x

    def draw_heightmap(self):
        heightmap = np.zeros(self.mapsize, dtype=np.uint8)
        for sector in self.level['sectors'].values():
            coords_DU = np.array(sector['vertices_xy'])  # Coordinates in DoomUnits (un-normalized)
            # skimage.draw needs all the coordinates to be > 0. They are centered and rescaled (1 pixel = 32DU)
            x, y = self._rescale_coord(coords_DU[:,0], coords_DU[:,1])
            px, py = draw.polygon(x,y, shape=tuple(self.mapsize))
            # 0 is for empty space
            color = self._rescale_value(sector['lump']['floor_height'], self.level['features']['floor_height_min'], 32) + 1
            heightmap[px, py] = color
            # since polygon only draws the inner part of the polygon we now draw the perimeter
            px, py = draw.polygon_perimeter(x, y, shape=tuple(self.mapsize))
            heightmap[px, py] = color
        return heightmap


    def draw_wallmap(self):
        vertices = self.level['lumps']['VERTEXES']
        wallmap = np.zeros(self.mapsize, dtype=np.uint8)
        for sector in self.level['sectors'].values():
            walls_linedefs = [line for line in sector['linedefs'] if line['left_sidedef'] == -1]
            for line in walls_linedefs:
                start = np.array([vertices[line['from']]['x'],vertices[line['from']]['y']]).astype(np.int32)
                end = np.array([vertices[line['to']]['x'],vertices[line['to']]['y']]).astype(np.int32)
                sx, sy = self._rescale_coord(start[0], start[1])
                ex, ey = self._rescale_coord(end[0], end[1])
                lx, ly = draw.line(sx, sy, ex, ey)
                wallmap[lx, ly] = 255
        return wallmap

    def draw_thingsmap(self):
        thingsmap = np.zeros(self.mapsize, dtype=np.uint16)
        things = self.level['lumps']['THINGS']
        for thing in things:
            tx, ty = self._rescale_coord(thing['x'], thing['y'])
            thingsmap[tx,ty] = thing['type']
        return thingsmap

    def compute_maps(self):
        self.mapsize_du = np.array([self.level['features']['width'], self.level['features']['height']])
        self.mapsize = np.ceil(self.mapsize_du / 32).astype(np.int32)

        # computing these maps require the knowledge of the level width and height
        self.level['features']['heightmap'] = self.draw_heightmap()
        self.level['features']['wallmap'] = self.draw_wallmap()
        self.level['features']['thingsmap'] = self.draw_thingsmap()
        self.level['features']['floormap'], self.level['features']['floors'] = label(self.level['features']['heightmap'], structure=np.ones((3,3)))
        pass

    def extract_features(self):
        # Computing the simplest set of features
        self.main_features()
        # TODO: It could be more convenient to rescale the image in DoomUnits, then rescaling everything down to the desired target output (<256)
        self.compute_maps()
        # topological features rely on computed maps
        self.topological_features()

        io.imshow(self.level['features']['heightmap'])
        io.show()
        io.imshow(self.level['features']['wallmap'])
        io.show()
        io.imshow(self.level['features']['floormap'])
        io.show()
        io.imshow(self.level['features']['thingsmap'])
        io.show()

        return self.level['features']


    def main_features(self):
        '''
        Extract features that are based on simple processing of data contained into the WAD file
        '''
        self.level['features'] = dict()
        self.level['features']['number_of_lines'] = len(self.level['lumps']['LINEDEFS'])
        self.level['features']['number_of_things'] = len(self.level['lumps']['THINGS'])
        self.level['features']['number_of_sectors'] = len(self.level['lumps']['SECTORS'])
        self.level['features']['number_of_subsectors'] = len(self.level['lumps']['SSECTORS'])
        self.level['features']['number_of_vertices'] = len(self.level['lumps']['VERTEXES'])
        self.level['features']['x_max'] = max(self.level['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        self.level['features']['y_max'] = max(self.level['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        self.level['features']['x_min'] = min(self.level['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        self.level['features']['y_min'] = min(self.level['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        self.level['features']['height'] = abs(self.level['features']['y_max']-self.level['features']['y_min'])+1
        self.level['features']['width'] = abs(self.level['features']['x_max']-self.level['features']['x_min'])+1
        self.level['features']['aspect_ratio'] = max(self.level['features']['height'], self.level['features']['width']) / min(self.level['features']['height'], self.level['features']['width'])

        floor_height = np.array([sector['lump']['floor_height'] for s_id, sector in self.level['sectors'].items()])
        ceiling_height = np.array([sector['lump']['ceiling_height'] for s_id, sector in self.level['sectors'].items()])
        room_height = ceiling_height-floor_height
        self.level['features']['floor_height_max'] = np.max(floor_height)
        self.level['features']['floor_height_min'] = np.min(floor_height)
        self.level['features']['floor_height_avg'] = np.mean(floor_height)
        self.level['features']['ceiling_height_max'] = np.max(ceiling_height)
        self.level['features']['ceiling_height_min'] = np.min(ceiling_height)
        self.level['features']['ceiling_height_avg'] = np.mean(ceiling_height)
        self.level['features']['room_height_max'] = np.max(room_height)
        self.level['features']['room_height_min'] = np.min(room_height)
        self.level['features']['room_height_avg'] = np.mean(room_height)


    def topological_features(self):
        self.level['features']['bounding_box_size'] = self.mapsize[0]*self.mapsize[1]
        self.level['features']['nonempty_size'] = np.count_nonzero(self.level['features']['heightmap'])
        # TODO: This should consider also nonwalkable things like decorations
        self.level['features']['walkable_area'] = self.level['features']['nonempty_size'] - np.count_nonzero(self.level['features']['wallmap'])
        self.level['features']['nonempty_percentage'] = self.level['features']['nonempty_size'] / self.level['features']['bounding_box_size']
        self.level['features']['walkable_percentage'] = self.level['features']['walkable_area'] / self.level['features']['nonempty_size']

import numpy as np
import skimage.draw as draw
from skimage import io
from scipy.ndimage.measurements import label
import WAD_Parser.Dictionaries.ThingTypes as ThingTypes

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
            if len(sector['vertices_xy'])==0:
                continue  # This sector is not referenced by any linedef so it's not a real sector
            coords_DU = np.array(sector['vertices_xy'])  # Coordinates in DoomUnits (un-normalized)
            # skimage.draw needs all the coordinates to be > 0. They are centered and rescaled (1 pixel = 32DU)
            x, y = self._rescale_coord(coords_DU[:,0], coords_DU[:,1])
            px, py = draw.polygon(x,y, shape=tuple(self.mapsize))
            # 0 is for empty space
            color = self._rescale_value(sector['lump']['floor_height'], self.level['features']['floor_height_min'], 32) + 1
            heightmap[px, py] = color
            # since polygon only draws the inner part of the polygon we now draw the perimeter
            if len(x) > 2 and len(y) > 2:
                px, py = draw.polygon_perimeter(x, y, shape=tuple(self.mapsize))
            heightmap[px, py] = color
        return heightmap


    def draw_wallmap(self):
        vertices = self.level['lumps']['VERTEXES']
        wallmap = np.zeros(self.mapsize, dtype=np.uint8)
        for sector in self.level['sectors'].values():
            if len(sector['vertices_xy'])==0:
                continue  # This sector is not referenced by any linedef so it's not a real sector
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
            is_unknown = ThingTypes.get_thing_category(thing['type']) == 'unknown'
            out_of_bounds = thing['x'] > self.level['features']['x_max'] or thing['x'] < self.level['features']['x_min'] or \
                            thing['y'] > self.level['features']['y_max'] or thing['x'] > self.level['features']['y_max']
            if is_unknown or out_of_bounds:
                continue
            tx, ty = self._rescale_coord(thing['x'], thing['y'])
            thingsmap[tx,ty] = thing['type']
        return thingsmap

    def compute_maps(self):
        self.mapsize_du = np.array([self.level['features']['width'], self.level['features']['height']])
        self.mapsize = np.ceil(self.mapsize_du / 32).astype(np.int32)

        # computing these maps require the knowledge of the level width and height
        self.level['maps']['heightmap'] = self.draw_heightmap()
        self.level['maps']['wallmap'] = self.draw_wallmap()
        self.level['maps']['thingsmap'] = self.draw_thingsmap()
        self.level['maps']['floormap'], self.level['features']['floors'] = label(self.level['maps']['heightmap'], structure=np.ones((3,3)))
        pass

    def extract_features(self):
        # Computing the simplest set of features
        self.main_features()
        self.compute_maps()
        # topological features rely on computed maps
        self.topological_features()


        return self.level['features'], self.level['maps']

    def _find_thing_category(self, category):
        found_things = np.in1d(self.level['maps']['thingsmap'], ThingTypes.get_category_things_types(category)).reshape(self.level['maps']['thingsmap'].shape)
        return np.where(found_things)

    def show_maps(self):
        io.imshow(self.level['maps']['wallmap'])
        io.show()
        io.imshow(self.level['maps']['thingsmap'])
        io.show()
        io.imshow(self.level['maps']['heightmap'])
        io.show()


    def _PolyArea(self, x, y):
        """
        Calculates the area of a polygon given its vertices
        :param x:
        :param y:
        :return: (ndarray) The polygon area
        """
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _bounding_box_dims(self, x, y):
        """
        Returns the height and width of the bounding box of a poligon specified by its vertices
        :param x:
        :param y:
        :return: (ndarray) [width, height]
        """
        xmax = np.max(x)
        ymax = np.max(y)
        xmin = np.min(x)
        ymin = np.min(y)
        return np.array([xmax - xmin + 1,ymax-ymin +1])

    def main_features(self):
        '''
        Extract features that are based on simple processing of data contained into the WAD file
        '''
        self.level['features'] = dict()
        self.level['maps'] = dict()
        self.level['features']['number_of_lines'] = len(self.level['lumps']['LINEDEFS']) if 'LINEDEFS' in self.level['lumps'] else 0
        self.level['features']['number_of_things'] = len(self.level['lumps']['THINGS']) if 'THINGS' in self.level['lumps'] else 0
        self.level['features']['number_of_sectors'] = len(self.level['lumps']['SECTORS']) if 'SECTORS' in self.level['lumps'] else 0
        self.level['features']['number_of_subsectors'] = len(self.level['lumps']['SSECTORS']) if 'SSECTORS' in self.level['lumps'] else 0
        self.level['features']['number_of_vertices'] = len(self.level['lumps']['VERTEXES'])
        self.level['features']['x_max'] = max(self.level['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        self.level['features']['y_max'] = max(self.level['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        self.level['features']['x_min'] = min(self.level['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        self.level['features']['y_min'] = min(self.level['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        self.level['features']['height'] = abs(self.level['features']['y_max']-self.level['features']['y_min'])+1
        self.level['features']['width'] = abs(self.level['features']['x_max']-self.level['features']['x_min'])+1
        self.level['features']['aspect_ratio'] = max(self.level['features']['height'], self.level['features']['width']) / min(self.level['features']['height'], self.level['features']['width'])


        floor_height = list()
        ceiling_height = list()
        sector_area = list()
        lines_per_sector = list()
        sector_aspect_ratio = list()
        for s_id, sector in self.level['sectors'].items():
            if len(sector['vertices_xy'])==0:
                continue  # This sector is not referenced by any linedef so it's not a real sector
            floor_height.append(sector['lump']['floor_height'])
            ceiling_height.append(sector['lump']['ceiling_height'])
            x, y = np.array(sector['vertices_xy'])[:,0],np.array(sector['vertices_xy'])[:,1]
            sector_area.append(self._PolyArea(x,y))
            lines_per_sector.append(len(sector['linedefs']))

            bounding_box_dim = self._bounding_box_dims(x,y)
            sector_aspect_ratio.append(np.max(bounding_box_dim)/np.min(bounding_box_dim))

        floor_height = np.array(floor_height)
        ceiling_height = np.array(ceiling_height)
        sector_area = np.array(sector_area)
        lines_per_sector = np.array(lines_per_sector)
        sector_aspect_ratio = np.array(sector_aspect_ratio)

        room_height = ceiling_height-floor_height
        self.level['features']['floor_height_max'] = float(np.max(floor_height))
        self.level['features']['floor_height_min'] = float(np.min(floor_height))
        self.level['features']['floor_height_avg'] = float(np.mean(floor_height))
        self.level['features']['ceiling_height_max'] = float(np.max(ceiling_height))
        self.level['features']['ceiling_height_min'] = float(np.min(ceiling_height))
        self.level['features']['ceiling_height_avg'] = float(np.mean(ceiling_height))
        self.level['features']['room_height_max'] = float(np.max(room_height))
        self.level['features']['room_height_min'] = float(np.min(room_height))
        self.level['features']['room_height_avg'] = float(np.mean(room_height))
        self.level['features']['sector_area_max'] = float(np.max(sector_area))
        self.level['features']['sector_area_min'] = float(np.min(sector_area))
        self.level['features']['sector_area_avg'] = float(np.mean(sector_area))
        self.level['features']['lines_per_sector_max'] = float(np.max(lines_per_sector))
        self.level['features']['lines_per_sector_min'] = float(np.min(lines_per_sector))
        self.level['features']['lines_per_sector_avg'] = float(np.mean(lines_per_sector))
        self.level['features']['sector_aspect_ratio_max'] = float(np.max(sector_aspect_ratio))
        self.level['features']['sector_aspect_ratio_min'] = float(np.min(sector_aspect_ratio))
        self.level['features']['sector_aspect_ratio_avg'] = float(np.mean(sector_aspect_ratio))



    def topological_features(self):
        self.level['features']['number_of_artifacts'] = int(np.size(self._find_thing_category('artifacts'), axis=-1))
        self.level['features']['number_of_powerups'] = int(np.size(self._find_thing_category('powerups'), axis=-1))
        self.level['features']['number_of_weapons'] = int(np.size(self._find_thing_category('weapons'), axis=-1))
        self.level['features']['number_of_ammunitions'] = int(np.size(self._find_thing_category('ammunitions'), axis=-1))
        self.level['features']['number_of_keys'] = int(np.size(self._find_thing_category('keys'), axis=-1))
        self.level['features']['number_of_monsters'] = int(np.size(self._find_thing_category('monsters'), axis=-1))
        self.level['features']['number_of_obstacles'] = int(np.size(self._find_thing_category('obstacles'), axis=-1))
        self.level['features']['number_of_decorations'] = int(np.size(self._find_thing_category('decorations'), axis=-1))


        self.level['features']['bounding_box_size'] = int(self.mapsize[0]*self.mapsize[1])
        self.level['features']['nonempty_size'] = int(np.count_nonzero(self.level['maps']['heightmap']))
        self.level['features']['walkable_area'] = int(self.level['features']['nonempty_size'] - np.count_nonzero(self.level['maps']['wallmap']) - self.level['features']['number_of_obstacles'])
        self.level['features']['nonempty_percentage'] = float(self.level['features']['nonempty_size'] / self.level['features']['bounding_box_size'])
        self.level['features']['walkable_percentage'] = float(self.level['features']['walkable_area'] / self.level['features']['nonempty_size'])

        start_location = self._find_thing_category('start')
        if not len(start_location[0]) or not len(start_location[1]):
            start_x, start_y = -1,-1
        else:
            start_x, start_y = start_location[0][0], start_location[1][0]
        self.level['features']['start_location_x_px'] = int(start_x)
        self.level['features']['start_location_y_px'] = int(start_y)

        self.level['features']['artifacts_per_walkable_area'] = float(self.level['features']['number_of_artifacts'] / self.level['features']['walkable_area'])
        self.level['features']['powerups_per_walkable_area'] = float(self.level['features']['number_of_powerups'] / self.level['features']['walkable_area'])
        self.level['features']['weapons_per_walkable_area'] = float(self.level['features']['number_of_weapons'] / self.level['features']['walkable_area'])
        self.level['features']['ammunitions_per_walkable_area'] = float(self.level['features']['number_of_ammunitions'] / self.level['features']['walkable_area'])
        self.level['features']['keys_per_walkable_area'] = float(self.level['features']['number_of_keys'] / self.level['features']['walkable_area'])
        self.level['features']['monsters_per_walkable_area'] = float(self.level['features']['number_of_monsters'] / self.level['features']['walkable_area'])
        self.level['features']['obstacles_per_walkable_area'] = float(self.level['features']['number_of_obstacles'] / self.level['features']['walkable_area'])
        self.level['features']['decorations_per_walkable_area'] = float(self.level['features']['number_of_decorations'] / self.level['features']['walkable_area'])

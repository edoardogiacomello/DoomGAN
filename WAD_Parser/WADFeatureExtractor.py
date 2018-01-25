import numpy as np
import skimage.draw as draw
from skimage import io
from scipy.ndimage.measurements import label
import scipy as sp
from skimage.measure import regionprops
import WAD_Parser.Dictionaries.ThingTypes as ThingTypes
import WAD_Parser.Dictionaries.LinedefTypes as LinedefTypes

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


    def draw_sector_maps(self):

        vertices = self.level['lumps']['VERTEXES']
        wallmap = np.zeros(self.mapsize, dtype=np.uint8)
        heightmap = np.zeros(self.mapsize, dtype=np.uint8)
        tagmap = np.zeros(self.mapsize, dtype=np.uint8)
        triggermap = np.zeros(self.mapsize, dtype=np.uint8)
        self.special_sector_map = np.zeros(self.mapsize, dtype=np.uint8)
        height_levels = sorted({s['lump']['floor_height'] for s in self.level['sectors'].values()})
        scale_color = lambda x, levels, a, b: int((levels.index(x)+1)*(b-a)/len(levels))
        for sector in self.level['sectors'].values():
            if len(sector['vertices_xy'])==0:
                continue  # This sector is not referenced by any linedef so it's not a real sector

            # HEIGHTMAP GENERATION
            coords_DU = np.array(sector['vertices_xy'])  # Coordinates in DoomUnits (un-normalized)
            # skimage.draw needs all the coordinates to be > 0. They are centered and rescaled (1 pixel = 32DU)
            x, y = self._rescale_coord(coords_DU[:, 0], coords_DU[:, 1])
            px, py = draw.polygon(x, y, shape=tuple(self.mapsize))
            # 0 is for empty space
            h = sector['lump']['floor_height']
            color = scale_color(h, height_levels, 0, 255)

            heightmap[px, py] = color
            # TAGMAP GENERATION (intermediate TRIGGERMAP: Sectors referenced by a trigger)
            tag = sector['lump']['tag']
            tag = tag if tag < 63 else 63
            tag = tag if tag >= 0 else 0
            tagmap[px, py] = tag
            # DAMAGING FLOOR MAP
            self.special_sector_map[px, py] = sector['lump']['special_sector']
            # since polygon only draws the inner part of the polygon we now draw the perimeter
            if len(x) > 2 and len(y) > 2:
                px, py = draw.polygon_perimeter(x, y, shape=tuple(self.mapsize))
            heightmap[px, py] = color

        # WALLMAP and TRIGGERMAP GENERATION
        for line in self.level['lumps']['LINEDEFS']:
            start = np.array([vertices[line['from']]['x'], vertices[line['from']]['y']]).astype(np.int32)
            end = np.array([vertices[line['to']]['x'], vertices[line['to']]['y']]).astype(np.int32)
            sx, sy = self._rescale_coord(start[0], start[1])
            ex, ey = self._rescale_coord(end[0], end[1])
            lx, ly = draw.line(sx, sy, ex, ey)
            if line['left_sidedef'] == -1: # If no sector on the other side
                # It's a wall
                wallmap[lx, ly] = 255
            linedef_type = LinedefTypes.get_index_from_type(line['types'])
            trigger = line['trigger']
            # clamping the trigger tag to [1,64] (otherwise the encoding will overflow)
            trigger = 64 if trigger > 64 else trigger
            if linedef_type != 0:  # Then something happens if this linedef is activated
                if trigger != 0:
                    if linedef_type in [32, 64]:
                            # Then the sectors tagged with "trigger" are remote doors if type is 32, or moving floors if 64
                            switch_color = 128+trigger-1  # The switch color is 128+i for i in [0,64)
                            sector_color = linedef_type+trigger-1  # The destination sector will be colored according to its type
                            # Drawing the switch
                            triggermap[lx, ly] = switch_color
                            # Coloring the dest sector (the remote door, the lift, etc)
                            dest_tag_pixels = np.transpose(np.where(tagmap==trigger))
                            for coord in dest_tag_pixels:
                                triggermap[tuple(coord)] = sector_color
                    if linedef_type in [192]:
                        # This is a teleporter source
                        # color the source
                        triggermap[lx, ly] = linedef_type + trigger - 1
                        # Finding and coloring the destination
                        # First, the destination sector is needed
                        destinations = [list(self._rescale_coord(t['x'], t['y'])) for t in self.level['lumps']['THINGS'] if t['type'] == 14] # 14 is thing type for "teleport landing"
                        dest_map = np.zeros(self.mapsize, dtype=np.bool)
                        for dest_coord in destinations:
                            dest_map[tuple(dest_coord)] = True
                        found_dest = np.where(np.logical_and((tagmap == trigger), dest_map))
                        # Color the destination
                        if len(found_dest):
                            triggermap[tuple(found_dest)] = linedef_type + trigger - 1

                if linedef_type in [10,12,14,16, 255]:
                    # It's a local door or an exit. Simply color the linedefs
                    triggermap[lx, ly] = linedef_type

        return wallmap, heightmap, triggermap

    def draw_thingsmap(self):
        thingsmap = np.zeros(self.mapsize, dtype=np.uint8)
        things = self.level['lumps']['THINGS']
        for thing in things:
            category = ThingTypes.get_category_from_type_id(thing['type'])
            is_unknown = category == 'unknown'
            out_of_bounds = thing['x'] > self.level['features']['x_max'] or thing['x'] < self.level['features']['x_min'] or \
                            thing['y'] > self.level['features']['y_max'] or thing['y'] < self.level['features']['y_min']
            if is_unknown or out_of_bounds:
                continue
            tx, ty = self._rescale_coord(thing['x'], thing['y'])
            if thingsmap[tx, ty] in ThingTypes.get_index_by_category('start'):
                # Avoid overwriting of player start location if something else is placed there (like a teleporter)
                continue
            thingsmap[tx,ty] = ThingTypes.get_index_from_type_id(thing['type'])

        return thingsmap

    def draw_textmap(self):
        """
               Represent the levels using 2-characters textual information.


               ENCODING:
               The encoding is in part taken from the TheVGLC dataset, so not all the information is displayed
               Each tile is represented as XY, with X being the VGLC encoding and Y being the ascii-encoding of the trigger tag.

               "-" : ["empty","out of bounds"],
               "X" : ["solid","wall"],
               "." : ["floor","walkable"],
               "," : ["floor","walkable","stairs"],
               "E" : ["enemy","walkable"],
               "W" : ["weapon","walkable"],
               "A" : ["ammo","walkable"],
               "H" : ["health","armor","walkable"],
               "B" : ["explosive barrel","walkable"],
               "K" : ["key","walkable"],
               "<" : ["start","walkable"],
               "T" : ["teleport","walkable","destination"],
               ":" : ["decorative","walkable"],
               "L" : ["door","locked"],
               "t" : ["teleport","source","activatable"],
               "+" : ["door","walkable","activatable"],
               ">" : ["exit","activatable"]

               :param json_db: 
               :param output_path: 
               :return: 
               """

        X_map = np.ndarray(shape=self.level['maps']['floormap'].shape, dtype=np.uint8)
        Y_map = np.ndarray(shape=self.level['maps']['floormap'].shape, dtype=np.uint8)
        txtmap = np.zeros(shape=(X_map.shape[0], X_map.shape[1] * 2), dtype=np.byte)

        from skimage.filters import roberts
        walls = self.level['maps']['wallmap'] != 0
        floor = self.level['maps']['floormap'] != 0
        change_in_height = roberts(self.level['maps']['heightmap']) != 0
        enemies = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('monsters'))
        weapons = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('weapons'))
        ammo = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('ammunitions'))
        health = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('powerups'))
        barrels = np.isin(self.level['maps']['thingsmap'], [ThingTypes.get_index_from_type_id(t) for t in [2035, 70]])
        keys = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('keys'))
        start = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('start'))
        teleport_dst = np.isin(self.level['maps']['thingsmap'], [ThingTypes.get_index_from_type_id(14)])
        decorative = np.isin(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category('decorations'))
        door_locked = np.isin(self.level['maps']['triggermap'],
                              [LinedefTypes.get_index_from_type(l) for l in [26, 28, 27, 32, 33, 34]])
        door_open = np.isin(self.level['maps']['triggermap'],
                            [LinedefTypes.get_index_from_type(l) for l in [1, 31, 46, 117, 118]] + list(range(32, 64)))
        teleport_src = np.logical_and(self.level['maps']['triggermap'] >= 192, self.level['maps']['triggermap'] < 255)
        exit = self.level['maps']['triggermap'] == 255

        dmging_floor = np.isin(self.special_sector_map, [4, 5, 7, 11, 16]) # If special tag is one of these, then it hurts the player


        # Rebuild the tagmap (contains sector and trigger tags)
        tagmap = np.where(self.level['maps']['triggermap'] >= 32, self.level['maps']['triggermap'], -1)  # -1 means "no tags here".
        tagmap = np.mod(tagmap + 1, 64)  # Now all tags are from 1 to 63, 0 means "no tag"

        X_map[...] = ord("-")
        Y_map[...] = ord("-") + tagmap
        Y_map = np.where(dmging_floor, ord("~"), Y_map)

        X_map = np.where(floor, ord("."), X_map)
        X_map = np.where(change_in_height, ord(","), X_map)
        X_map = np.where(decorative, ord(":"), X_map)
        X_map = np.where(enemies, ord("E"), X_map)
        X_map = np.where(weapons, ord("W"), X_map)
        X_map = np.where(ammo, ord("A"), X_map)
        X_map = np.where(health, ord("H"), X_map)
        X_map = np.where(barrels, ord("B"), X_map)
        X_map = np.where(keys, ord("K"), X_map)
        X_map = np.where(start, ord("<"), X_map)
        X_map = np.where(teleport_dst, ord("T"), X_map)
        X_map = np.where(teleport_src, ord("t"), X_map)
        X_map = np.where(door_locked, ord("L"), X_map)
        X_map = np.where(door_open, ord("+"), X_map)
        X_map = np.where(walls, ord("X"), X_map)
        X_map = np.where(exit, ord(">"), X_map)



        txtmap[:, 0::2] = X_map
        txtmap[:, 1::2] = Y_map
        return txtmap.tolist()

            
    def compute_maps(self):
        self.mapsize_du = np.array([self.level['features']['width'], self.level['features']['height']])
        self.mapsize = np.ceil(self.mapsize_du / 32).astype(np.int32)

        # computing these maps require the knowledge of the level width and height
        #tag_map is an intermediate map needed to build the trigger map
        self.level['maps']['wallmap'], self.level['maps']['heightmap'], self.level['maps']['triggermap'] = self.draw_sector_maps()
        self.level['maps']['thingsmap'] = self.draw_thingsmap()
        enumerated_floors, self.level['features']['floors'] = label(self.level['maps']['heightmap'], structure=np.ones((3,3)))
        self.level['maps']['floormap'] = ((enumerated_floors > 0)*255).astype(np.uint8)

        self.level['text'] = self.draw_textmap()

    def _features_to_scalar(self):
        for feature in self.level['features']:
            try:
                self.level['features'][feature] = np.asscalar(self.level['features'][feature])
            except AttributeError:
                pass

    def extract_features(self):
        # Computing the simplest set of features
        self.main_features()
        self.compute_maps()
        # topological features rely on computed maps
        self.image_features()
        # Convert every feature to scalar
        self._features_to_scalar()
        return self.level['features'], self.level['maps'], self.level['text']

    def _find_thing_category(self, category):
        found_things = np.in1d(self.level['maps']['thingsmap'], ThingTypes.get_index_by_category(category)).reshape(self.level['maps']['thingsmap'].shape)
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



    def image_features(self):
        # Creating auxiliary feature maps
        nonempty_map = self.level['maps']['floormap'].astype(np.bool).astype(np.uint8)
        floormap = self.level['maps']['floormap']
        walkablemap = np.logical_and(nonempty_map, np.logical_not(self.level['maps']['wallmap'])).astype(np.uint8)
        # Computing features
        features = regionprops(nonempty_map)
        features_floors = regionprops(floormap)
        feature_walkablemap = regionprops(walkablemap)
        region_props = ['area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number',
                        'extent', 'filled_area', 'major_axis_length', 'minor_axis_length', 'orientation',
                        'perimeter', 'solidity']
        for prop in region_props:
            # Adding global features
            self.level['features']["level_{}".format(prop)] = features[0][prop]

            # Adding also statistics for features computed at each floor
            if prop not in ['bbox_area']:  # (bbox_area is always the global one)
                prop_floors_vector = [features_floors[f][prop] for f in range(len(features_floors))]
                self.level['features']["floors_{}_mean".format(prop)] = sp.mean(prop_floors_vector)
                self.level['features']["floors_{}_min".format(prop)] = sp.amin(prop_floors_vector)
                self.level['features']["floors_{}_max".format(prop)] = sp.amax(prop_floors_vector)
                self.level['features']["floors_{}_std".format(prop)] = sp.std(prop_floors_vector)

        # Adding hu moments and centroid
        for i in range(7):
            self.level['features']["level_hu_moment_{}".format(i)] = features[0]['moments_hu'][i]
        self.level['features']["level_centroid_x"] = features[0]['centroid'][0]
        self.level['features']["level_centroid_y"] = features[0]['centroid'][1]


        self.level['features']['number_of_artifacts'] = int(np.size(self._find_thing_category('artifacts'), axis=-1))
        self.level['features']['number_of_powerups'] = int(np.size(self._find_thing_category('powerups'), axis=-1))
        self.level['features']['number_of_weapons'] = int(np.size(self._find_thing_category('weapons'), axis=-1))
        self.level['features']['number_of_ammunitions'] = int(
            np.size(self._find_thing_category('ammunitions'), axis=-1))
        self.level['features']['number_of_keys'] = int(np.size(self._find_thing_category('keys'), axis=-1))
        self.level['features']['number_of_monsters'] = int(np.size(self._find_thing_category('monsters'), axis=-1))
        self.level['features']['number_of_obstacles'] = int(np.size(self._find_thing_category('obstacles'), axis=-1))
        self.level['features']['number_of_decorations'] = int(
            np.size(self._find_thing_category('decorations'), axis=-1))

        self.level['features']['walkable_area'] = feature_walkablemap[0]['area']
        self.level['features']['walkable_percentage'] = float(
            feature_walkablemap[0]['area'] / self.level['features']['level_area'])

        start_location = self._find_thing_category('start')
        if not len(start_location[0]) or not len(start_location[1]):
            start_x, start_y = -1, -1
            print("This level has no explicit start location")
        else:
            start_x, start_y = start_location[0][0], start_location[1][0]
        self.level['features']['start_location_x_px'] = int(start_x)
        self.level['features']['start_location_y_px'] = int(start_y)

        self.level['features']['artifacts_per_walkable_area'] = float(
            self.level['features']['number_of_artifacts'] / self.level['features']['walkable_area'])
        self.level['features']['powerups_per_walkable_area'] = float(
            self.level['features']['number_of_powerups'] / self.level['features']['walkable_area'])
        self.level['features']['weapons_per_walkable_area'] = float(
            self.level['features']['number_of_weapons'] / self.level['features']['walkable_area'])
        self.level['features']['ammunitions_per_walkable_area'] = float(
            self.level['features']['number_of_ammunitions'] / self.level['features']['walkable_area'])
        self.level['features']['keys_per_walkable_area'] = float(
            self.level['features']['number_of_keys'] / self.level['features']['walkable_area'])
        self.level['features']['monsters_per_walkable_area'] = float(
            self.level['features']['number_of_monsters'] / self.level['features']['walkable_area'])
        self.level['features']['obstacles_per_walkable_area'] = float(
            self.level['features']['number_of_obstacles'] / self.level['features']['walkable_area'])
        self.level['features']['decorations_per_walkable_area'] = float(
            self.level['features']['number_of_decorations'] / self.level['features']['walkable_area'])


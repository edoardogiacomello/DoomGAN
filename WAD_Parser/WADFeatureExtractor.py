import skimage.draw as draw
from skimage import io
from skimage.morphology import label
import scipy as sp
from skimage.measure import regionprops
import WAD_Parser.Dictionaries.ThingTypes as ThingTypes
import WAD_Parser.Dictionaries.LinedefTypes as LinedefTypes
from WAD_Parser.RoomTopology import *

class WADFeatureExtractor(object):
    """
    Class for extracting a set of feature from a given WAD level or a set of Feature Maps
    """
       


    def _rescale_coord(self, x_du, y_du, wad_features, factor=32):
        x_centered = x_du - wad_features['x_min']
        y_centered = y_du - wad_features['y_min']
        x = np.floor(x_centered / factor).astype(np.int32)
        y = np.floor(y_centered / factor).astype(np.int32)
        return x, y


    def draw_sector_maps(self, level_dict, mapsize_px, wad_features):

        vertices = level_dict['lumps']['VERTEXES']
        wallmap = np.zeros(mapsize_px, dtype=np.uint8)
        heightmap = np.zeros(mapsize_px, dtype=np.uint8)
        tagmap = np.zeros(mapsize_px, dtype=np.uint8)
        triggermap = np.zeros(mapsize_px, dtype=np.uint8)
        self.special_sector_map = np.zeros(mapsize_px, dtype=np.uint8)
        height_levels = sorted({s['lump']['floor_height'] for s in level_dict['sectors'].values()})
        scale_color = lambda x, levels, a, b: int((levels.index(x)+1)*(b-a)/len(levels))
        for sector in level_dict['sectors'].values():
            if len(sector['vertices_xy'])==0:
                continue  # This sector is not referenced by any linedef so it's not a real sector

            # HEIGHTMAP GENERATION
            coords_DU = np.array(sector['vertices_xy'])  # Coordinates in DoomUnits (un-normalized)
            # skimage.draw needs all the coordinates to be > 0. They are centered and rescaled (1 pixel = 32DU)
            x, y = self._rescale_coord(coords_DU[:, 0], coords_DU[:, 1], wad_features)
            px, py = draw.polygon(x, y, shape=tuple(mapsize_px))
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
                px, py = draw.polygon_perimeter(x, y, shape=tuple(mapsize_px))
            heightmap[px, py] = color

        # WALLMAP and TRIGGERMAP GENERATION
        for line in level_dict['lumps']['LINEDEFS']:
            start = np.array([vertices[line['from']]['x'], vertices[line['from']]['y']]).astype(np.int32)
            end = np.array([vertices[line['to']]['x'], vertices[line['to']]['y']]).astype(np.int32)
            sx, sy = self._rescale_coord(start[0], start[1],wad_features)
            ex, ey = self._rescale_coord(end[0], end[1], wad_features)
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
                        destinations = [list(self._rescale_coord(t['x'], t['y'], wad_features)) for t in level_dict['lumps']['THINGS'] if t['type'] == 14] # 14 is thing type for "teleport landing"
                        dest_map = np.zeros(mapsize_px, dtype=np.bool)
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

    def draw_thingsmap(self, level_dict, wad_features, mapsize_px):
        thingsmap = np.zeros(mapsize_px, dtype=np.uint8)
        things = level_dict['lumps']['THINGS']
        for thing in things:
            category = ThingTypes.get_category_from_type_id(thing['type'])
            is_unknown = category == 'unknown'
            out_of_bounds = thing['x'] > wad_features['x_max'] or thing['x'] < wad_features['x_min'] or \
                            thing['y'] > wad_features['y_max'] or thing['y'] < wad_features['y_min']
            if is_unknown or out_of_bounds:
                continue
            tx, ty = self._rescale_coord(thing['x'], thing['y'], wad_features)
            if thingsmap[tx, ty] in ThingTypes.get_index_by_category('start'):
                # Avoid overwriting of player start location if something else is placed there (like a teleporter)
                continue
            thingsmap[tx,ty] = ThingTypes.get_index_from_type_id(thing['type'])

        return thingsmap

    def draw_textmap(self, level_dict):
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

        X_map = np.ndarray(shape=level_dict['maps']['floormap'].shape, dtype=np.uint8)
        Y_map = np.ndarray(shape=level_dict['maps']['floormap'].shape, dtype=np.uint8)
        txtmap = np.zeros(shape=(X_map.shape[0], X_map.shape[1] * 2), dtype=np.byte)

        from skimage.filters import roberts
        walls = level_dict['maps']['wallmap'] != 0
        floor = level_dict['maps']['floormap'] != 0
        change_in_height = roberts(level_dict['maps']['heightmap']) != 0
        enemies = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('monsters'))
        weapons = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('weapons'))
        ammo = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('ammunitions'))
        health = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('powerups'))
        barrels = np.isin(level_dict['maps']['thingsmap'], [ThingTypes.get_index_from_type_id(t) for t in [2035, 70]])
        keys = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('keys'))
        start = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('start'))
        teleport_dst = np.isin(level_dict['maps']['thingsmap'], [ThingTypes.get_index_from_type_id(14)])
        decorative = np.isin(level_dict['maps']['thingsmap'], ThingTypes.get_index_by_category('decorations'))
        door_locked = np.isin(level_dict['maps']['triggermap'],
                              [LinedefTypes.get_index_from_type(l) for l in [26, 28, 27, 32, 33, 34]])
        door_open = np.isin(level_dict['maps']['triggermap'],
                            [LinedefTypes.get_index_from_type(l) for l in [1, 31, 46, 117, 118]] + list(range(32, 64)))
        teleport_src = np.logical_and(level_dict['maps']['triggermap'] >= 192, level_dict['maps']['triggermap'] < 255)
        exit = level_dict['maps']['triggermap'] == 255

        dmging_floor = np.isin(self.special_sector_map, [4, 5, 7, 11, 16]) # If special tag is one of these, then it hurts the player


        # Rebuild the tagmap (contains sector and trigger tags)
        tagmap = np.where(level_dict['maps']['triggermap'] >= 32, level_dict['maps']['triggermap'], -1)  # -1 means "no tags here".
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

            
    def compute_maps(self, level_dict, wad_features):
        maps=dict()
        
        mapsize_du = np.array([wad_features['width'], wad_features['height']])
        mapsize_px = np.ceil(mapsize_du / 32).astype(np.int32)

        # computing these maps require the knowledge of the level width and height
        #tag_map is an intermediate map needed to build the trigger map
        maps['wallmap'], maps['heightmap'], maps['triggermap'] = self.draw_sector_maps(level_dict, mapsize_px, wad_features)
        maps['thingsmap'] = self.draw_thingsmap(level_dict, wad_features)
        enumerated_floors, wad_features['floors'] = label(maps['heightmap']>0, connectivity=2, return_num=True)
        maps['floormap'] = enumerated_floors.astype(np.uint8)

        level_dict['text'] = self.draw_textmap(level_dict)
        return maps

    def _features_to_scalar(self, level_dict):
        for feature in level_dict['features']:
            try:
                level_dict['features'][feature] = np.asscalar(level_dict['features'][feature])
            except AttributeError:
                pass


    def extract_features_from_maps(self, floormap, wallmap, thingsmap, feature_names=None):
        """
        Extracts features given images of each sample. If feature_names is None then a dictionary containing all the possible features is returned.
        Otherwise, an array is returned
        :param floormap: The floormap of the level
        :param wallmap: The Wallmap of the level
        :param thingsmap: The thingsmap of the level
        :param feature_names: [None] List of feature names to select. If not none, the returned result is an array.
        :return: A dictionary containing feature values, or an array with values corresponding to feature_names.
        """
        features = dict()
        if wallmap is None:
            wallmap = np.zeros_like(floormap)
        if thingsmap is None:
            thingsmap = np.zeros_like(floormap)
        png_features = self.image_features(floormap=floormap,
                                           wallmap=wallmap,
                                           thingsmap=thingsmap)
        _, _, graph_metrics = topological_features(floormap, prepare_for_doom=False)
        features.update(png_features)
        features.update(graph_metrics)
        if feature_names is None:
            return features
        else:
            return np.asarray([features[name] for name in feature_names])


    def extract_features_from_wad(self, level_dict):
        """
        :param level_dict: Wad in dictionary format as produced by WADReader.read()
        :return: 
        """
        
        level_dict['features'] = dict()
        level_dict['maps'] = dict()
        # Computing the simplest set of features
        wad_feats = self.wad_features(level_dict=level_dict)
        level_dict['features'].update(wad_feats)
        feature_maps = self.compute_maps(level_dict, level_dict['features'])
        level_dict['maps'].update(feature_maps)
        # morphological features rely on computed maps
        png_features = self.image_features(floormap=level_dict['maps']['floormap'], wallmap=level_dict['maps']['wallmap'], thingsmap=level_dict['maps']['thingsmap'])
        level_dict['features'].update(png_features)
        # "Flattening" the floormap to a binary image
        level_dict['maps']['floormap'] = ((level_dict['maps']['floormap'] > 0) * 255).astype(np.uint8)
        # Computing topological features and maps
        level_dict['maps']['roommap'], level_dict['graph'], metrics = topological_features(level_dict['maps']['floormap'], prepare_for_doom=False)
        level_dict['features'].update(metrics)
        # Convert every feature to scalar
        self._features_to_scalar(level_dict)
        return level_dict['features'], level_dict['maps'], level_dict['text'], level_dict['graph']

    def _find_thing_category(self, category, thingsmap):
        found_things = np.in1d(thingsmap, ThingTypes.get_index_by_category(category)).reshape(thingsmap.shape)
        return np.where(found_things)



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

    def wad_features(self, level_dict):
        '''
        Returns a dict of features that are based on data contained into the WAD file.
        :param level_dict: The dictionary representation of the level as produced by the WADReader
        '''
        feature_dict = dict()
        feature_dict['number_of_lines'] = len(level_dict['lumps']['LINEDEFS']) if 'LINEDEFS' in level_dict['lumps'] else 0
        feature_dict['number_of_things'] = len(level_dict['lumps']['THINGS']) if 'THINGS' in level_dict['lumps'] else 0
        feature_dict['number_of_sectors'] = len(level_dict['lumps']['SECTORS']) if 'SECTORS' in level_dict['lumps'] else 0
        feature_dict['number_of_subsectors'] = len(level_dict['lumps']['SSECTORS']) if 'SSECTORS' in level_dict['lumps'] else 0
        feature_dict['number_of_vertices'] = len(level_dict['lumps']['VERTEXES'])
        feature_dict['x_max'] = max(level_dict['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        feature_dict['y_max'] = max(level_dict['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        feature_dict['x_min'] = min(level_dict['lumps']['VERTEXES'], key=lambda v: v['x'])['x']
        feature_dict['y_min'] = min(level_dict['lumps']['VERTEXES'], key=lambda v: v['y'])['y']
        feature_dict['height'] = abs(feature_dict['y_max']-feature_dict['y_min'])+1
        feature_dict['width'] = abs(feature_dict['x_max']-feature_dict['x_min'])+1
        feature_dict['aspect_ratio'] = max(feature_dict['height'], feature_dict['width']) / min(feature_dict['height'], feature_dict['width'])


        floor_height = list()
        ceiling_height = list()
        sector_area = list()
        lines_per_sector = list()
        sector_aspect_ratio = list()
        for s_id, sector in level_dict['sectors'].items():
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
        feature_dict['floor_height_max'] = float(np.max(floor_height))
        feature_dict['floor_height_min'] = float(np.min(floor_height))
        feature_dict['floor_height_avg'] = float(np.mean(floor_height))
        feature_dict['ceiling_height_max'] = float(np.max(ceiling_height))
        feature_dict['ceiling_height_min'] = float(np.min(ceiling_height))
        feature_dict['ceiling_height_avg'] = float(np.mean(ceiling_height))
        feature_dict['room_height_max'] = float(np.max(room_height))
        feature_dict['room_height_min'] = float(np.min(room_height))
        feature_dict['room_height_avg'] = float(np.mean(room_height))
        feature_dict['sector_area_max'] = float(np.max(sector_area))
        feature_dict['sector_area_min'] = float(np.min(sector_area))
        feature_dict['sector_area_avg'] = float(np.mean(sector_area))
        feature_dict['lines_per_sector_max'] = float(np.max(lines_per_sector))
        feature_dict['lines_per_sector_min'] = float(np.min(lines_per_sector))
        feature_dict['lines_per_sector_avg'] = float(np.mean(lines_per_sector))
        feature_dict['sector_aspect_ratio_max'] = float(np.max(sector_aspect_ratio))
        feature_dict['sector_aspect_ratio_min'] = float(np.min(sector_aspect_ratio))
        feature_dict['sector_aspect_ratio_avg'] = float(np.mean(sector_aspect_ratio))
        return feature_dict


    def image_features(self, floormap, wallmap, thingsmap):
        """ Returns a dict containing the set of features that are based on the floormap and wallmaps of the level """
        feature_dict = dict()
        
        # Creating auxiliary feature maps
        nonempty_map = floormap.astype(np.bool).astype(np.uint8)
        walkablemap = np.logical_and(nonempty_map, np.logical_not(wallmap)).astype(np.uint8)
        # Computing features
        features = regionprops(nonempty_map)
        features_floors = regionprops(floormap)
        feature_walkablemap = regionprops(walkablemap)
        region_props = ['area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number',
                        'extent', 'filled_area', 'major_axis_length', 'minor_axis_length', 'orientation',
                        'perimeter', 'solidity']
        for prop in region_props:
            # Adding global features
            feature_dict["level_{}".format(prop)] = features[0][prop]

            # Adding also statistics for features computed at each floor
            if prop not in ['bbox_area']:  # (bbox_area is always the global one)
                prop_floors_vector = [features_floors[f][prop] for f in range(len(features_floors))]
                feature_dict["floors_{}_mean".format(prop)] = sp.mean(prop_floors_vector)
                feature_dict["floors_{}_min".format(prop)] = sp.amin(prop_floors_vector)
                feature_dict["floors_{}_max".format(prop)] = sp.amax(prop_floors_vector)
                feature_dict["floors_{}_std".format(prop)] = sp.std(prop_floors_vector)

        # Adding hu moments and centroid
        for i in range(7):
            feature_dict["level_hu_moment_{}".format(i)] = features[0]['moments_hu'][i]
        feature_dict["level_centroid_x"] = features[0]['centroid'][0]
        feature_dict["level_centroid_y"] = features[0]['centroid'][1]


        feature_dict['number_of_artifacts'] = int(np.size(self._find_thing_category('artifacts', thingsmap), axis=-1))
        feature_dict['number_of_powerups'] = int(np.size(self._find_thing_category('powerups', thingsmap), axis=-1))
        feature_dict['number_of_weapons'] = int(np.size(self._find_thing_category('weapons', thingsmap), axis=-1))
        feature_dict['number_of_ammunitions'] = int(
            np.size(self._find_thing_category('ammunitions', thingsmap), axis=-1))
        feature_dict['number_of_keys'] = int(np.size(self._find_thing_category('keys', thingsmap), axis=-1))
        feature_dict['number_of_monsters'] = int(np.size(self._find_thing_category('monsters', thingsmap), axis=-1))
        feature_dict['number_of_obstacles'] = int(np.size(self._find_thing_category('obstacles', thingsmap), axis=-1))
        feature_dict['number_of_decorations'] = int(
            np.size(self._find_thing_category('decorations', thingsmap), axis=-1))

        feature_dict['walkable_area'] = feature_walkablemap[0]['area']
        feature_dict['walkable_percentage'] = float(
            feature_walkablemap[0]['area'] / feature_dict['level_area'])

        start_location = self._find_thing_category('start', thingsmap)
        if not len(start_location[0]) or not len(start_location[1]):
            start_x, start_y = -1, -1
        else:
            start_x, start_y = start_location[0][0], start_location[1][0]
        feature_dict['start_location_x_px'] = int(start_x)
        feature_dict['start_location_y_px'] = int(start_y)

        feature_dict['artifacts_per_walkable_area'] = float(
            feature_dict['number_of_artifacts'] / feature_dict['walkable_area'])
        feature_dict['powerups_per_walkable_area'] = float(
            feature_dict['number_of_powerups'] / feature_dict['walkable_area'])
        feature_dict['weapons_per_walkable_area'] = float(
            feature_dict['number_of_weapons'] / feature_dict['walkable_area'])
        feature_dict['ammunitions_per_walkable_area'] = float(
            feature_dict['number_of_ammunitions'] / feature_dict['walkable_area'])
        feature_dict['keys_per_walkable_area'] = float(
            feature_dict['number_of_keys'] / feature_dict['walkable_area'])
        feature_dict['monsters_per_walkable_area'] = float(
            feature_dict['number_of_monsters'] / feature_dict['walkable_area'])
        feature_dict['obstacles_per_walkable_area'] = float(
            feature_dict['number_of_obstacles'] / feature_dict['walkable_area'])
        feature_dict['decorations_per_walkable_area'] = float(
            feature_dict['number_of_decorations'] / feature_dict['walkable_area'])
        return feature_dict

import itertools
from struct import *
from WAD_Parser import Lumps
from WAD_Parser.WADFeatureExtractor import WADFeatureExtractor
import re, os
from skimage import io
from skimage.measure import find_contours
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

# Data specification taken from http://www.gamers.org/dhs/helpdocs/dmsp1666.html
# Implementation by Edoardo Giacomello Nov - 2017



class LumpInfo(dict):
    def __init__(self, filepos=None, size=None, name=None):
        """

        :param filepos:  An integer holding a pointer to the start of the lump's data in the file.
        :param size: An integer representing the size of the lump in bytes.
        :param name: A 8 byte encoded ascii string, eventually padded with 00
        """
        super()
        self['filepos'] = filepos
        self['size'] = size
        self['name'] = name


    def from_bytes(self, byte_stream):
        self['filepos'], = unpack("i", byte_stream[0:4])
        self['size'], = unpack("i", byte_stream[4:8])
        self['name'] = Lumps.decode_doomstring(byte_stream[8:16])
        return self

    def to_bytes(self):
        info_bytes = bytearray()
        info_bytes += pack("i", self['filepos'])
        info_bytes += pack("i", self['size'])
        info_bytes += Lumps.encode_doomstring(self['name'])
        return info_bytes
class WAD(dict):
    def __init__(self, mode):
        """
            Dictionary structured representation of a WAD file. Fields that are dictionary keys are unprocessed data from
            the file itself, while object attributes are "secondary access keys" (structures not directly encoded in the
             WAD file but built for faster access to data.)

             mode: 'R' for reading or 'W' for writing.

            Example:
                self['lumps'] contains the list of all the lumps.
                        Lumps describing levels are processed as list or dict(s), the others are kept as raw bytes.
                self['directory'] is the list of lump info, the structure reflects the one encoded in the WAD file.
                self.levels contains the lumps grouped by each level. (secondary key, not directly encoded into the file)
                self.sectors contains the linedefs sorrounding each map sector (secondary key)
            Warning:
                It's suggested to use the WADReader and WADWriter class in order to read and write a WAD.
                If you need to edit a WAD object, please consider copying it into another WAD() using from_bytes and
                to_bytes methods.
        """
        super()
        self['header'] = {
                'identification' : 'PWAD',  # Ascii identifier: IWAD or PWAD
                'numlumps' : 0, # An integer specifying the number of lumps in the WAD.
                'infotableofs' : 0 # An integer holding a pointer to the location of the directory.
            }
        self['lumps'] = []  # List of lumps, some processed, other in byte format
        self['directory'] = list() # List of lumpinfo
        self.levels = [] # this division in levels is not part of the wad but it's done for fast access
        self.map_regex = re.compile('MAP\d\d?')
        self.em_regex = re.compile('E\d*M\d\d?')
        self.errors = list()

        self.mode = mode
        self.current_lump_offset = 12  # Keeps track of the offset in bytes of the last. The header is always 12 bytes long



    def from_bytes(self, byte_stream):
        '''
        Builds a WAD object from the byte stream from a .WAD file.
        :param byte_stream:
        :return:
        '''
        assert self.mode == 'R', "Cannot read a WAD opened in write mode. " \
                                 "Please consider copying your WAD() into a new one " \
                                 "using to_bytes and from_bytes methods"
        try:
            self['header']['identification'] = Lumps.decode_doomstring(byte_stream[0:4])
            self['header']['numlumps'], = unpack("i", byte_stream[4:8])
            self['header']['infotableofs'], = unpack("i", byte_stream[8:12])

            # the pattern for grouped record is
            # [byte[start:start+length] for start in range(offset, offset+n_items*length, length)]
            lump_info_records = [byte_stream[start:start+16] for start in range(self['header']['infotableofs'],
                                                                                self['header']['infotableofs']
                                                                                +self['header']['numlumps']*16, 16)]
            # Populate the lump directory
            for lump_info_bytes in lump_info_records:
                lumpinfo = LumpInfo().from_bytes(lump_info_bytes)
                self['directory'].append(lumpinfo)

            # Parsing lumps
            for lump in self['directory']:
                if lump['size'] < 0:
                    self.errors.append({'object': lump, 'description': 'Negative size lump', 'fatal':False})
                    # Some files are corrupted and have a negative lump size. They'd cause a segfault if launched with doom
                    # We try to go on extracting as much data as we can from the WAD file.
                    continue
                lumpname = lump['name']
                # If the lump is a level descriptor, then create a new secondary key for the level
                if (self.map_regex.match(lump['name']) is not None) or (self.em_regex.match(lump['name']) is not None):
                    self.levels.append({'name':lumpname, 'lumps':{}})

                lump_bytes = byte_stream[lump['filepos']:lump['filepos'] + lump['size']]

                if lumpname in Lumps.known_lumps_classes.keys() and len(lump_bytes) > 0:
                    # Got a level lump and need to parse it...
                    l = Lumps.known_lumps_classes[lumpname]().from_bytes(lump_bytes)
                    if len(self.levels)>0:  #  otherwise we have found a level lump before the level description, which should not happen
                        self.levels[-1]['lumps'][lumpname] = l

                    # Adding processed lump to the lump list
                    self['lumps'].append(l)
                else:
                    # got an empty lump or another type of lump (such textures etc) that is not useful.
                    # Adding raw format to the lump list
                    self['lumps'].append(lump_bytes)

            # Cleaning empty levels (some wad files has random level descriptor with no lumps following them
            for l in self.levels:
                if 'SECTORS' not in l['lumps'] or 'LINEDEFS' not in l['lumps']:
                    self.levels.remove(l)


            # Building other secondary access keys
            # levels[sector][sector_id] = {sector: lump, sidedefs: list(lump), linedefs: list(lump), vertices=list(), vertex_path=list()}
            # Retrieving linedefs for each sector
            for level in self.levels:
                level['sectors'] = {}
                # This part of code makes the access to sectors and vertices easier.
                # Lines, Vertices, Sidedef and Sectors are indexed by three lists, and they are connected in this way:
                # Line -> Vertices, Line -> Sidedef(s) -> Sector

                # Create an entry for each sector.
                for sec_id, sec_lump in enumerate(level['lumps']['SECTORS']):
                    level['sectors'][sec_id] = {'lump': sec_lump, 'linedefs': list(), 'sidedefs': list(), 'vertex_path':list(), 'vertices_xy':list()}

                # For each linedef, fill the corresponding sector record(s)
                for linedef_id, linedef_lump in enumerate(level['lumps']['LINEDEFS']):
                    r_side_id = linedef_lump['right_sidedef']
                    r_sidedef = level['lumps']['SIDEDEFS'][r_side_id]
                    r_sector = r_sidedef['sector']
                    level['sectors'][r_sector]['linedefs'].append(linedef_lump)
                    level['sectors'][r_sector]['sidedefs'].append(r_sidedef)

                    l_side_id = linedef_lump['left_sidedef']
                    if l_side_id != -1:
                        l_sidedef = level['lumps']['SIDEDEFS'][l_side_id]
                        l_sector = l_sidedef['sector']
                        level['sectors'][l_sector]['linedefs'].append(linedef_lump)
                        level['sectors'][l_sector]['sidedefs'].append(l_sidedef)

                # create vertex path for each sector for drawing
                for sector_id, sector in level['sectors'].items():
                    # Make the graph G(Linedefs, Verices) undirected
                    edges = set()
                    for linedef in sector['linedefs']:
                        if (linedef['from'] != linedef['to']):  # Avoids single-vertex linedefs
                            edges.add((linedef['from'],linedef['to']))
                            edges.add((linedef['to'],linedef['from']))
                    if len(edges) > 0:   # Avoid crashes if some sectors are empty
                        # "hops" is the list of vertex indices as visited by a drawing algorithm
                        hops = list()
                        next_edge = min(edges)

                        if next_edge[0] not in hops:
                            hops.append(next_edge[0])
                        if next_edge[1] not in hops:
                            hops.append(next_edge[1])
                        while (len(edges) > 1):
                            edges.remove((next_edge[1], next_edge[0]))
                            edges.remove((next_edge[0], next_edge[1]))
                            next_edges = set(filter(lambda x: x[0] == hops[-1] or x[1] == hops[-1], edges))
                            if len(next_edges) == 0:
                                break
                            possible_next = min(next_edges)
                            if possible_next[1] == hops[-1]:
                                next_edge = (possible_next[1], possible_next[0])
                            else:
                                next_edge = possible_next
                            if next_edge[-1] not in hops:
                                hops.append(next_edge[-1])
                        sector['vertex_path'] = hops
                        sector['vertices_xy'] = [(level['lumps']['VERTEXES'][v_id]['x'], level['lumps']['VERTEXES'][v_id]['y']) for v_id in hops]
        except Exception as e:
            # All known exceptions found in the database are avoided, this exception will catch everything else that is unexpected
            # and will produce a fatal error
            self.errors = list()
            self.errors.append({'object': self, 'description': e, 'fatal':True})

        return self

    def add_lump(self, lumpname, lump):
        """
        Adds a new lump named lumpname and updates the information in the directory. Increments the current_lump_offset.
        :param lumpname: lump name. It will be converted in doomstring format.
        :param lump: a @Lumps object, or None for level descriptors or other zero-sized lumps.
        :return: None
        """
        assert self.mode == 'W', "Cannot write a WAD opened in read mode. " \
                                 "Please consider copying your WAD() into a new one " \
                                 "using to_bytes and from_bytes methods"
        if lump is None:
            lump_bytes = bytes()
        else:
            lump_bytes = lump.to_bytes()
        size = len(lump_bytes)
        self['directory'].append(LumpInfo(filepos=self.current_lump_offset, size=size, name=lumpname))
        self['lumps'].append(lump_bytes)
        # Updating directory and header information
        self.current_lump_offset += size
        self['header']['numlumps'] += 1
        # The infotableoffset is always kept at the end of the file
        self['header']['infotableofs'] = self.current_lump_offset

    def to_bytes(self):
        # Build the entire file
        # header to bytes
        wad_bytes = bytearray()
        wad_bytes += bytes('PWAD', encoding='ascii')
        wad_bytes += pack('i', self['header']['numlumps'])
        wad_bytes += pack('i', self['header']['infotableofs'])

        # Adding Lumps
        for lump in self['lumps']:
            wad_bytes += lump

        # Adding directory
        for lumpinfo in self['directory']:
            wad_bytes += lumpinfo.to_bytes()
        return wad_bytes


class WADWriter(object):
    def __init__(self):
        """
        Class for writing a WAD file.
        Start by defining a new level with add_level(), then place new sectors and "things". Changes are submitted only
         on save or on the addition of a new level, since some optimization are due (duplicated vertices check, etc).
        """
        # due to the way the WAD works, for keeping track of the lumps size for filling the dictionary the byte representation is
        # needed. But for checking duplicated vertices/linedefs/etc it would be needed to convert back each lump before the
        # check. For avoiding this problem, a set of lumps is stored in the writer and written only when the level is
        # fully specified.
        self.wad = WAD('W')
        self.current_level = None
        self.lumps = {'THINGS':Lumps.Things(), 'LINEDEFS':Lumps.Linedefs(), 'VERTEXES':Lumps.Vertexes(),'SIDEDEFS': Lumps.Sidedefs(), 'SECTORS':Lumps.Sectors()}  # Temporary lumps for this level


    def from_images(self, floormap, wallmap, thingsmap=None, level_coord_scale = 64, debug = False):
        floormap = io.imread(floormap).astype(dtype=np.bool)
        # Pad with a frame for getting boundaries
        floormap = np.pad(floormap, pad_width=(1, 1), mode='constant')
        wallmap = io.imread(wallmap).astype(dtype=np.bool)
        # Pad with a frame for getting boundaries
        wallmap = np.pad(wallmap, pad_width=(1, 1), mode='constant')
        thingsmap = io.imread(thingsmap).astype(np.uint16)
        thingsmap = np.pad(thingsmap, pad_width=(1, 1), mode='constant')

        floormap = morphology.binary_dilation(floormap)

        self.add_level()

        # Apply some morphology transformation to the maps for combining them and removing artifacts
        denoised = morphology.remove_small_holes(floormap, min_size=16)
        denoised = morphology.remove_small_objects(denoised, min_size=16)
        denoised_walls = np.logical_and(denoised, np.logical_not(wallmap))
        cleaned_walls = morphology.remove_small_holes(denoised_walls, min_size=16)
        cleaned_walls = morphology.remove_small_objects(cleaned_walls, min_size=16)


        if debug:
            # Display the image and plot all contours found
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 5), sharex=True,
                                   sharey=True, subplot_kw={'adjustable': 'box-forced'})

        # Placing sectors, one floor at a time

        from scipy.ndimage.measurements import label
        segmented, total_floors = label(cleaned_walls, structure=np.ones((3, 3)))
        floors = [np.equal(segmented, f) for f in range(1, total_floors)]
        for floor_id, floor in enumerate(floors):
            contours_walls = find_contours(floor, 0.5, positive_orientation='low')
            for i, contour in enumerate(contours_walls):
                ax[4].plot(contour[:, 1], contour[:, 0], linewidth=1)
                vertices = (contour * level_coord_scale).astype(np.int).tolist()
                sector_id = self.add_sector(vertices, ceiling_height=128, tag=floor_id)

            # Placing teleporters
            if total_floors > 1:
                unreached = list(range(len(floors)))
                # Teleporters are needed. Place a landing somewhere in the sector
                possible_coords = np.transpose(np.nonzero(floor))
                rand_indices = tuple(np.random.choice(range(len(possible_coords)), size=2, replace=False).tolist())
                dest_coord = possible_coords[rand_indices[0]] * level_coord_scale
                src_coord = possible_coords[rand_indices[1]] * level_coord_scale
                dest_sector = floor_id
                while dest_sector == floor_id:
                    dest_sector = int(np.random.choice(unreached, replace=False))
                unreached.remove(dest_sector)
                # Add a destination in this sector
                self.add_teleporter_destination(dest_coord[0], dest_coord[1])
                # Add a source for another sector
                self.add_teleporter_source(src_coord[0], src_coord[1], dest_sector, inside=sector_id)

        if debug:
            ax[0].imshow(floormap, cmap=plt.cm.gray)
            ax[1].imshow(wallmap, cmap=plt.cm.gray)
            ax[2].imshow(denoised, cmap=plt.cm.gray)
            ax[3].imshow(denoised_walls, cmap=plt.cm.gray)
            ax[4].imshow(cleaned_walls, cmap=plt.cm.gray)
            ax[4].imshow(segmented, cmap=plt.cm.gray)
            plt.show()

        # Finding and placing new player start
        possible_pos = np.where(np.logical_and(np.equal(thingsmap,1), cleaned_walls))
        if (len(possible_pos[0]) == 0):
            possible_pos = np.where(cleaned_walls)
        rand_i = np.random.choice(range(len(possible_pos[0])))
        start_pos = (int(possible_pos[0][rand_i])*level_coord_scale, int(possible_pos[1][rand_i])*level_coord_scale)
        self.set_start(start_pos[0], start_pos[1])


    def add_teleporter_destination(self, x, y):
        self.add_thing(x, y, thing_type=14)

    def add_teleporter_source(self, x, y, to_sector, inside, size=32):
        x=int(x)
        y=int(y)
        to_sector=int(to_sector)
        halfsize=size//2
        vertices = [(x-halfsize, y+halfsize),(x+halfsize, y+halfsize),(x+halfsize, y-halfsize),(x-halfsize, y-halfsize)]

        self.add_sector(vertices, floor_flat='GATE1', wall_texture='-', linedef_flags=4, linedef_type=97, linedef_trigger=to_sector, inside=inside)



    def set_start(self, x, y):
        self.lumps['THINGS'].add_thing(int(x), int(y), angle=0, type=1, options=0)

    def add_thing(self, x,y, thing_type, options=7, angle=0):
        self.lumps['THINGS'].add_thing(int(x), int(y), angle=angle, type=thing_type, options=options)

    def add_sector(self, vertices_coords, floor_height=-8, ceiling_height=256, floor_flat='FLOOR0_1', ceiling_flat='FLOOR4_1', lightlevel=128, special=0, tag=0, wall_texture='BRONZE1', linedef_type=0, linedef_flags=1, linedef_trigger=0, inside=None):

        # In order to add a sector one must add:
        # A vertex for each vertex in the sector, only if not already present
        # a linedef for each edge, if not already present
        # a sidedef for each linedef, put on the correct side of the linedef
        # the sector lump itself

        # Create a sector
        sector_id = self.lumps['SECTORS'].add_sector(floor_height, ceiling_height, floor_flat, ceiling_flat, lightlevel, special, tag)


        # Create and lookup vertices
        vertices_id = list()
        for v in vertices_coords:
            v_id = self.lumps['VERTEXES'].add_vertex(v)
            vertices_id.append(v_id)

        # Iterate over (v0, v1), (v1, v2), ..., (vn-1, vn).
        # Adding the first element at the end for closing the polygon
        startiter, enditer = itertools.tee(vertices_id+[vertices_id[0]], 2)
        next(enditer, None)  # shift the end iterator by one
        for start, end in zip(startiter, enditer):
            # Create a new sidedef
            sidedef_id = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0, upper_texture='-', lower_texture='-',
                                               middle_texture=wall_texture, sector=sector_id)
            if inside is not None:
                left_sidedef = self.lumps['SIDEDEFS'].add_sidedef(x_offset=0, y_offset=0, upper_texture='-', lower_texture='-',
                                               middle_texture=wall_texture, sector=inside)
            else:
                left_sidedef=-1
            # Linedef creation/Lookup
            self.lumps['LINEDEFS'].add_linedef(start, end, flags=linedef_flags, types=linedef_type, trigger=linedef_trigger, right_sidedef_index=sidedef_id, left_sidedef_index=left_sidedef)
        return sector_id

    def _commit_level(self):
        """
        Writes the set of lumps to the WAD object
        :return:
        """
        assert self.current_level is not None, "Cannot write a level with an empty name"
        # Create a new level descriptor in the lump directory
        self.wad.add_lump(self.current_level, None)
        # Add the lumps to WAD file
        self.wad.add_lump('THINGS', self.lumps['THINGS'])
        self.wad.add_lump('LINEDEFS', self.lumps['LINEDEFS'])
        self.wad.add_lump('SIDEDEFS', self.lumps['SIDEDEFS'])
        self.wad.add_lump('VERTEXES', self.lumps['VERTEXES'])
        self.wad.add_lump('SECTORS', self.lumps['SECTORS'])
        self.lumps = {'THINGS':Lumps.Things(), 'LINEDEFS':Lumps.Linedefs(), 'VERTEXES':Lumps.Vertexes(),'SIDEDEFS': Lumps.Sidedefs(), 'SECTORS':Lumps.Sectors()}

    def add_level(self, name='MAP01'):
        # If it's not the first level, then commit the previous
        if self.current_level is not None:
            self._commit_level()
        self.current_level = name

    def save(self, fp):
        # Always commit the last level
        self._commit_level()
        wad_bytes = self.wad.to_bytes()
        with open(fp,'wb') as out:
            out.write(wad_bytes)
        print('Calling ZenNode...')
        subprocess.check_call(["bsp", fp, '-o', fp] )



class WADReader(object):
    """"Batch reader for WAD files"""

    def read(self, w):
        """
        Reads a wad file representing it as a dictionary
        :param w: path of the .WAD file
        :return:
        """
        with open(w, 'rb') as file:
            wad_name = w.split('/')[-1]
            wad = WAD('R').from_bytes(file.read())
            record = {'wad_name': wad_name, 'wad': wad, 'errors':list()}
            if len(record['wad'].errors) > 0:
                if (not record['wad'].errors[0]['fatal']):
                    print("{}: Malformed file, results may be altered".format(w))
                else:
                    print("{}: Fatal error in file structure, skipping file.".format(w))
                    record['errors'] += record['wad'].errors
        return record

    def save_sample(self, wad, path, root_path = '', wad_info=None):
        """
        Saves the wad maps (as .png) and features (as .json) to the "path" folder for each level in the wad.
        Also adds the produced file paths to the level features,
        if root_path is set then these paths are relative to that folder instead of being absolute paths
        if wad_info is not None, then adds its fields as features
        :param wad: the parsed wad file to save as feature maps
        :param path: the output folder
        :param root_path: the path to which the paths stored in the features should be relative to
        :return: None
        """
        os.makedirs(path, exist_ok=True)
        for level in wad['levels']:
            base_filename=path+wad['wad_name'].split('.')[-2]+'_'+level['name']
            # Path relative to the dataset root that will be stored in the database
            relative_path = base_filename.replace(root_path, '')
            # Adding the features
            for map in level['maps']:
                # Adding the corresponding path as feature for further access
                level['features']['path_{}'.format(map)] = relative_path + '_{}.png'.format(map)
                io.imsave(base_filename + '_{}.png'.format(map), level['maps'][map])
            for wadinfo in wad_info:
                # Adding wad info (author, etc) to the level features.
                if wadinfo not in level['features']:  # Computed features have priority over passed features
                    level['features'][wadinfo] = wad_info[wadinfo]
            # Completing the features with the level slot
            level['features']['slot'] = level['name']
            # Doing the same for the other features
            level['features']['path_json'] = relative_path + '.json'
            with open(base_filename + '.json', 'w') as jout:
                json.dump(level['features'], jout)



    def extract(self, wad_fp, save_to=None, root_path=None, update_record=None):
        """
        Compute the image representation and the features of each level contained in the wad file.
        If 'save_to' is set, then also do the following:
            - saves a json file for each level inside the 'save_to' folder
            - saves the set of maps as png images inside the 'save_to' folder
            - adds the relative path of the above mentioned files as level features
        Morover, if 'save_to' is set, then you may want to specify a 'root_path' for avoiding to save the full path in the features.
        If 'update_record' is set to a json dictionary (perhaps containing info about the wad author, title, etc),
        then this function stores all the update_record fields into the level features dictionary.
        :return: Parsed Wad
        """
        parsed_wad = self.read(wad_fp)
        for error in parsed_wad['errors']:
            if error['fatal']:
                return None
        parsed_wad['levels'] = list()
        for level in parsed_wad['wad'].levels:
            extractor = WADFeatureExtractor(level)
            features, maps = extractor.extract_features()
            parsed_wad['levels'] += [{'name': level['name'], 'features': features, 'maps':maps}]
        if save_to is not None:
            self.save_sample(parsed_wad, save_to, root_path, update_record)
        return parsed_wad



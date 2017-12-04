from collections import namedtuple
import binascii
import itertools
from struct import *
from WAD_Parser import Lumps
from WAD_Parser.WADFeatureExtractor import WADFeatureExtractor
import re, os
from skimage import io
from skimage.measure import find_contours
import subprocess

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

    def from_image(self, image):
        """
        Creates a wad file from an image representation
        :param image:
        :return:
        """
        # TODO: Implement this
        image = io.imread(image)

        import matplotlib.pyplot as plt

        from skimage import measure


        # Find contours at a constant value of 0.8
        contours = find_contours(image, 0.5)

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        #ax.imshow(image)
        from skimage.measure import approximate_polygon
        for n, contour in enumerate(contours):
            #ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            coords = approximate_polygon(contour, tolerance=2.5)
            ax.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)




        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

        pass

    def set_start(self, x, y):
        self.lumps['THINGS'].add_thing(x, y, angle=0, type=1, options=0)

    def add_thing(self, x,y, thing_type, options, angle=0):
        self.lumps['THINGS'].add_thing(x, y, angle=angle, type=thing_type, options=options)

    def add_sector(self, vertices_coords, floor_height=-8, ceiling_height=256, floor_flat='FLOOR0_1', ceiling_flat='FLOOR4_1', lightlevel=128, special=0, tag=0, wall_texture='BRONZE1'):

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
            # Linedef creation/Lookup
            self.lumps['LINEDEFS'].add_linedef(start, end, flags=1, types=0, trigger=0, right_sidedef_index=sidedef_id)


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

    def save_sample(self, wad, path):
        os.makedirs(path, exist_ok=True)
        for level in wad['levels']:
            base_filename=path+wad['wad_name'].split('.')[-2]+'_'+level['name']
            # Feature maps will be saved as images, all the other will be stored in a json file
            io.imsave(base_filename+'_floormap.png', level['maps']['floormap'])
            io.imsave(base_filename+'_heightmap.png', level['maps']['heightmap'])
            io.imsave(base_filename+'_thingsmap.png', level['maps']['thingsmap'])
            io.imsave(base_filename+'_wallmap.png', level['maps']['wallmap'])
            import json
            with open(base_filename+'.json', 'w') as jout:
                json.dump(level['features'], jout)
        pass


    def extract(self, w, save_to=None):
        """
        Compute the image representation and the features of each level contained in the wad file.
        If 'save_to' is set, then saves the features to the given folder
        :return:
        """
        parsed_wad = self.read(w)
        for error in parsed_wad['errors']:
            if error['fatal']:
                return None
        parsed_wad['levels'] = list()
        for level in parsed_wad['wad'].levels:
            extractor = WADFeatureExtractor(level)
            features, maps = extractor.extract_features()
            # extractor.show_maps()
            parsed_wad['levels'] += [{'name': level['name'], 'features': features, 'maps':maps}]
        if save_to is not None:
            self.save_sample(parsed_wad, save_to)
        return parsed_wad

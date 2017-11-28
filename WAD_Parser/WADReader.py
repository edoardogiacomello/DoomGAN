from collections import namedtuple
import binascii
from struct import *
from WAD_Parser import Lumps
from WAD_Parser.WADFeatureExtractor import WADFeatureExtractor

# Data specification taken from http://www.gamers.org/dhs/helpdocs/dmsp1666.html
# Implementation by Edoardo Giacomello Nov - 2017






class LumpInfo(dict):
    def __init__(self):
        super()
        self['filepos'] = None  # An integer holding a pointer to the start of the lump's data in the file.
        self['size'] = None  # An integer representing the size of the lump in bytes.
        self['name'] = None  # A 8 byte encoded ascii string, eventually padded with 00

    def from_bytes(self, byte_stream):
        self['filepos'], = unpack("i", byte_stream[0:4])
        self['size'], = unpack("i", byte_stream[4:8])
        self['name'] = byte_stream[8:16].decode('ascii')
        return self

class WAD(dict):
    def __init__(self):
        """
            Dictionary structured representation of a WAD file. Fields that are dictionary keys are unprocessed data from
            the file itself, while object attributes are "secondary access keys" (structures not directly encoded in the
             WAD file but built for faster access to data.)
            Example:
                self['lumps'] contains the list of all the lumps.
                        Lumps describing levels are processed as list or dict(s), the others are kept as raw bytes.
                self['directory'] is the list of lump info, the structure reflects the one encoded in the WAD file.
                self.levels contains the lumps grouped by each level. (secondary key, not directly encoded into the file)
                self.sectors contains the linedefs sorrounding each map sector (secondary key)
            Warning:
                Avoid directly alter to "secondary access key" data, since the behaviour is not granted to be
                 consistent when writing modification to file. Always use proper methods for storing data in the file.
        """
        super()
        self['header'] = {
                'identification' : None,  # Ascii identifier: IWAD or PWAD
                'numlumps' : None, # An integer specifying the number of lumps in the WAD.
                'infotableofs' : None # An integer holding a pointer to the location of the directory.
            }
        self['lumps'] = []  # List of lumps, some processed, other in byte format
        self['directory'] = list() # List of lumpinfo
        self.levels = [] # this division in levels is not part of the wad but it's done for fast access

    def from_bytes(self, byte_stream):
        '''
        Builds a WAD object from the byte stream from a .WAD file.
        :param byte_stream:
        :return:
        '''

        self['header']['identification'] = byte_stream[0:4].decode('ascii')
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

            # Remove zero padding from the lump name and extract the corresponding bytes
            lumpname = lump['name'].replace('\x00','')
            lump_bytes = byte_stream[lump['filepos']:lump['filepos'] + lump['size']]

            # If the lump is a level descriptor, then create a new secondary key for the level
            if lump['size'] == 0 and (lumpname.startswith('MAP') or (lumpname[0]=='E' and lumpname[2]=='M')):
                self.levels.append({'name':lumpname, 'lumps':{}})

            if lumpname in Lumps.known_lumps_classes.keys():
                # Got a level lump and need to parse it...
                l = Lumps.known_lumps_classes[lumpname]().from_bytes(lump_bytes)
                self.levels[-1]['lumps'][lumpname] = l
                # Adding processed lump to the lump list
                self['lumps'].append(l)
            else:
                # got another type of lump (such textures etc) that is not useful. Add raw format to the lump list
                self['lumps'].append(lump_bytes)

        # Building other secondary access keys
        # levels[sector][sector_id] = {sector: lump, sidedefs: list(lump), linedefs: list(lump), vertices=list(), vertex_path=list()}
        # Retrieving linedefs for each sector
        for level in self.levels:
            level['sectors'] = {}
            # Create an entry for each sector
            for sec_id, sec_lump in enumerate(level['lumps']['SECTORS']):
                level['sectors'][sec_id] = {'lump': sec_lump, 'linedefs': list(), 'sidedefs': list(), 'vertices': dict(), 'vertex_path':list()}

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
                    edges.add((linedef['from'],linedef['to']))
                    edges.add((linedef['to'],linedef['from']))
                # "hops" is the list of vertex indices as visited by a drawing algorithm
                hops = list()
                next_edge = min(edges)
                hops.append(next_edge[0])
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
                    hops.append(next_edge[-1])
                sector['vertex_path'] = hops
                sector['vertices_xy'] = [(level['lumps']['VERTEXES'][v_id]['x'], level['lumps']['VERTEXES'][v_id]['y']) for v_id in hops]
        return self



class WADReader(object):
    """"Batch reader for WAD files"""


    def __init__(self, wad_files):
        self.wad_files = wad_files
        self.features = list()  # List of feature dict for each level
        self.parsed_wads = list()  # List of parsed wads, each with structure: {'wad_name', 'wad', 'levels':[{'name', 'images', 'features'}]}
        self.read()


    def read(self):
        """
        Reads the list of wad file representing them as a list of dictionaries
        :return: None
        """
        for w in self.wad_files:
            with open(w, 'rb') as file:
                wad_name = w.split('/')[-1]
                wad = WAD().from_bytes(file.read())
                record = {'wad_name': wad_name, 'wad': wad}
                self.parsed_wads.append(record)

    def extract(self):
        """
        Compute the image representation and the features of each level contained in the wad file
        :return:
        """
        for w in self.parsed_wads:
            for level in w['wad'].levels:
                features = WADFeatureExtractor(level).extract_features()
                level['features'] = features
                w['levels'] = {'name': level['name'], 'features': features}
        return self.parsed_wads

data = WADReader(['/home/edoardo/Projects/DoomPCGML/dataset/CommunityLevels/WADs/DoomII/Original/24hour_a_24HOURS.WAD']).extract()
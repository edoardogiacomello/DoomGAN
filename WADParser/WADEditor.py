from collections import namedtuple
import binascii
from struct import *
import Lumps
import re

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
       super()
       self['header'] = {
               'identification' : None,  # Ascii identifier: IWAD or PWAD
               'numlumps' : None, # An integer specifying the number of lumps in the WAD.
               'infotableofs' : None # An integer holding a pointer to the location of the directory.
           }
       self['lumps'] = list() # List of lumps
       self['directory'] = list() # List of lumpinfo

    def from_bytes(self, byte_stream):

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

        current_level = None
        # Parse lumps
        for lump in self['directory']:
            l = None
            if lump['size'] == 0 and (  'MAP' in lump['name'] or re.match('E.M..', lump['name']) is not None):
                # FIXME: This makes the debugger crash
                current_level = lump
            l = Lumps.Things().from_bytes(lump) if 'THINGS' in lump['name'] else l

        return self



class WadEditor(object):
    """"Batch editor for WAD files"""


    def __init__(self, wad_files):
        self.wad_files = wad_files

    def read(self):
        for w in self.wad_files:
            with open(w, 'rb') as file:
                self.parsed = WAD().from_bytes(file.read())



WadEditor(['/home/edoardo/Projects/DoomPCGML/dataset/CommunityLevels/WADs/DoomII/Original/zukky_zukky.wad']).read()
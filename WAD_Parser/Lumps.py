from struct import *

class Things(list):
    def __init__(self):
        super()


    def from_bytes(self, byte_stream):
        # Each thing is 10 bytes long
        things_bytes = [byte_stream[s:s+10] for s in range(0, len(byte_stream), 10)]
        for tb in things_bytes:
            thing = {}
            thing["x"], = unpack('h',tb[0:2])
            thing["y"], = unpack('h',tb[2:4])
            thing["angle"], = unpack('h',tb[4:6])
            thing["type"], = unpack('h',tb[6:8])
            thing["options"], = unpack('h',tb[8:10]) # Difficulty level for which the thing is present
            self.append(thing)
        return self


    def add_thing(self, x, y, angle, type, options):
        self.append({'x': x, 'y': y, 'angle': angle, 'type': type, 'options': options})


    def to_bytes(self):
        lump_bytes = bytearray()
        for thing in self:
            lump_bytes += thing["x"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += thing["y"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += thing["angle"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += thing["type"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += thing["options"].to_bytes(2, byteorder='little', signed=True)
        return lump_bytes


class Linedefs(list):
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each linedef is 14 bytes long
        linedef_bytes = [byte_stream[s:s + 14] for s in range(0, len(byte_stream), 14)]
        if len(linedef_bytes[-1])<14:
            del linedef_bytes[-1]
        for lb in linedef_bytes:
            linedef={}
            linedef['from'], = unpack('h',lb[0:2]) # From this vertex
            linedef['to'], = unpack('h', lb[2:4])
            linedef['flags'], = unpack('h', lb[4:6])
            linedef['types'], = unpack('h', lb[6:8])
            linedef['trigger'], = unpack('h', lb[8:10])
            linedef['right_sidedef'], = unpack('h', lb[10:12]) # mandatory, direction must comply
            linedef['left_sidedef'], = unpack('h', lb[12:14]) # or -1 (FFFF) if there's no sector on the left
            self.append(linedef)
        return self

    def add_linedef(self, vertex_from, vertex_to, flags, types, trigger, right_sidedef_index, left_sidedef_index=-1):
        """
        Add a new linedef. If the linedef in the opposite direction is already present, then
        the old linedef is used and the right_sidedef_index is applied to the left_sidedef.
        If it's present in the same direction then the linedef is added as new.
        """
        found_left = [self.index(l) for l in self if l['from']==vertex_to and l['to']==vertex_from]
        if found_left:
            self[found_left[0]]['flags'] = flags
            self[found_left[0]]['types'] = types
            self[found_left[0]]['trigger'] = trigger
            self[found_left[0]]['left_sidedef'] = right_sidedef_index
        else:
            line = {'from': vertex_from, 'to': vertex_to, 'flags': flags, 'types': types, 'trigger': trigger,
                    'right_sidedef': right_sidedef_index, 'left_sidedef': left_sidedef_index}
            self.append(line)







    def to_bytes(self):
        lump_bytes = bytearray()
        for linedef in self:
            lump_bytes += linedef["from"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["to"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["flags"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["types"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["trigger"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["right_sidedef"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += linedef["left_sidedef"].to_bytes(2, byteorder='little', signed=True)
        return lump_bytes


class Sidedefs(list):
    # What texture to draw along a LINEDEF
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each sidedef's record is 30 bytes
        sidedef_bytes = [byte_stream[s:s + 30] for s in range(0, len(byte_stream), 30)]
        for sb in sidedef_bytes:
            sidedef = dict()
            sidedef['x_offset'], = unpack('h',sb[0:2])
            sidedef['y_offset'], = unpack('h',sb[2:4])
            sidedef['upper_texture'] = decode_doomstring(sb[4:12])  # '-' for simple sidedefs
            sidedef['lower_texture'] = decode_doomstring(sb[12:20])  # '-' for simple sidedefs
            sidedef['middle_texture'] = decode_doomstring(sb[20:28]) # normal or full texture, '-' if transparent
            sidedef['sector'], = unpack('h',sb[28:30]) # Sector the sidedef is facing
            self.append(sidedef)
        # Texture names are from TEXTURE1/2 resources. Wall patches into the directory are referenced through the PNAMES lump
        return self

    def add_sidedef(self, x_offset, y_offset, upper_texture, lower_texture, middle_texture, sector):
        """
        Adds a new sidedef and return its id
        :param x_offset:
        :param y_offset:
        :param upper_texture:
        :param lower_texture:
        :param middle_texture:
        :param sector:
        :return:
        """
        self.append({ 'x_offset': x_offset, 'y_offset': y_offset, 'upper_texture': upper_texture, 'lower_texture': lower_texture, 'middle_texture': middle_texture, 'sector': sector})
        return len(self)-1

    def to_bytes(self):
        lump_bytes = bytearray()
        for sidedef in self:
            lump_bytes += sidedef["x_offset"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += sidedef["y_offset"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += encode_doomstring(sidedef["upper_texture"])
            lump_bytes += encode_doomstring(sidedef["lower_texture"])
            lump_bytes += encode_doomstring(sidedef["middle_texture"])
            lump_bytes += sidedef["sector"].to_bytes(2, byteorder='little', signed=True)
        return lump_bytes

class Vertexes(list):
    # Even though the correct name should be "vertices" the standard lump name is used
    def __init__(self):
        """
        Starting and ending points for LINEDEFS and SEGS
        """
        super()
    def from_bytes(self, byte_stream):
        # each record is 4 bytes
        vertexes_bytes = [byte_stream[s:s + 4] for s in range(0, len(byte_stream), 4)]
        for vb in vertexes_bytes:
            vertex = dict()
            vertex['x'], = unpack('h', vb[0:2])
            vertex['y'], = unpack('h', vb[2:4])
            self.append(vertex)
        return self

    def add_vertex(self, v):
        """
        Adds a new vertex to the list and returns its index. If the vertex is already present, then just return its index
        :param v: Tuple (x, y)
        :return: int, the vertex index
        """
        v = {'x': v[0], 'y': v[1]}
        if v not in self:
            self.append(v)
            return len(self)-1
        return self.index(v)

    def to_bytes(self):
        lump_bytes = bytearray()
        for vertex in self:
            lump_bytes += vertex["x"].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += vertex["y"].to_bytes(2, byteorder='little', signed=True)
        return lump_bytes


class Segs(list):
    """"
    The SEGS are stored in a sequential order determined by the SSECTORS,
    which are part of the NODES recursive tree.
    """
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each seg is 12 bytes in 6 <short> fields
        segs_bytes = [byte_stream[s:s + 12] for s in range(0, len(byte_stream), 12)]
        for sb in segs_bytes:
            seg = {}
            seg['start'], = unpack('h', sb[0:2])  # start_vertex
            seg['end'], = unpack('h', sb[2:4])
            seg['angle'], = unpack('h', sb[4:6])  # 0= east, 16384=north, -16384=south, -32768=west.
            #  This is also know as BAMS for Binary Angle Measurement
            seg['linedef'], = unpack('h', sb[6:8])
            seg['direction'], = unpack('h', sb[8:10]) # 0 if seg and linedef go the same direction, 1 oth.
            # 0 if seg is on the RIGHT side of linedef, 1 otherwise
            seg['offset'], = unpack('h', sb[10:12])  # distance along the linedef to the start of this seg
            self.append(seg)
        return self

    #### SINCE SEGS, SSECTORS AND NODES ARE COMPUTED VIA AN EXTERNAL TOOL, NO WRITING FUNCTIONS HAVE BEEN WRITTEN FOR NOW


class SSectors(list):
    """
    SSECTOR stands for sub-sector. These divide up all the SECTORS into
    convex polygons. They are then referenced through the NODES resources.
    There will be (number of nodes + 1) ssectors.
      Each ssector is 4 bytes in 2 <short> fields:
    
    (1) This many SEGS are in this SSECTOR...
    (2) ...starting with this SEG number
    
      The segs in ssector 0 should be segs 0 through x, then ssector 1
    contains segs x+1 through y, ssector 2 containg segs y+1 to z, etc. """

    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        ssector_bytes = [byte_stream[s:s + 4] for s in range(0, len(byte_stream), 4)]
        for sb in ssector_bytes:
            subsector = dict()
            subsector['segs_count'], = unpack('h', sb[0:2])
            subsector['start_seg'], = unpack('h', sb[2:4])
            self.append(subsector)
        return self

    #### SINCE SEGS, SSECTORS AND NODES ARE COMPUTED VIA AN EXTERNAL TOOL, NO WRITING FUNCTIONS HAVE BEEN WRITTEN FOR NOW

class Nodes(list):
    """
    The NODES are branches in a binary space partition (BSP) that divides
    up the level and is used to determine which walls are in front of others,
    a process know as hidden-surface removal. The SSECTORS (sub-sectors) and
    SEGS (segments) lumps are necessary parts of the structure.
      A BSP tree is normally used in 3d space, but DOOM uses a simplified
    2d version of the scheme. Basically, the idea is to keep dividing the
    map into smaller spaces until each of the smallest spaces contains only
    wall segments which cannot possibly occlude (block from view) other
    walls in its own space. The smallest, undivided spaces will become
    SSECTORS. Each wall segment is part or all of a linedef (and thus a
    straight line), and becomes a SEG. All of the divisions are kept track
    of in a binary tree structure, which is used to greatly speed the
    rendering process (drawing what is seen).
    """
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each node is 28 bytes in 14 <short> fields:
        nodes_bytes = [byte_stream[s:s + 28] for s in range(0, len(byte_stream), 28)]
        for nb in nodes_bytes:
            node = dict()
            node['x'], = unpack('h', nb[ 0: 2])
            node['y'], = unpack('h', nb[ 2: 4])
            node['dx'], = unpack('h', nb[ 4: 6])
            node['dy'], = unpack('h', nb[ 6: 8])
            node['yubr'], = unpack('h', nb[ 8:10])
            node['ylbr'], = unpack('h', nb[10:12])
            node['xlbr'], = unpack('h', nb[12:14])
            node['xubr'], = unpack('h', nb[14:16])
            node['yubl'], = unpack('h', nb[16:18])
            node['ylbl'], = unpack('h', nb[18:20])
            node['xlbl'], = unpack('h', nb[20:22])
            node['xubl'], = unpack('h', nb[22:24])
            node['rchild'], = unpack('h', nb[24:26])
            node['lchild'], = unpack('h', nb[26:28])
            self.append(node)
        return self

        #### SINCE SEGS, SSECTORS AND NODES ARE COMPUTED VIA AN EXTERNAL TOOL, NO WRITING FUNCTIONS HAVE BEEN WRITTEN FOR NOW

class Sectors(list):
    """
    A SECTOR is a horizontal (east-west and north-south) area of the map
    where a floor height and ceiling height is defined. It can have any
    shape. Any change in floor or ceiling height or texture requires a
    new sector (and therefore separating linedefs and sidedefs). If you
    didn't already know, this is where you find out that DOOM is in many
    respects still a two-dimensional world, because there can only be ONE
    floor height in each sector. No buildings with two floors, one above
    the other, although fairly convincing illusions are possible.
    """
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each sector's record is 26 bytes, comprising 2 <short> fields, 
        # then 2 <8-byte string> fields, then 3 <short> fields:
        sector_bytes = [byte_stream[s:s + 26] for s in range(0, len(byte_stream), 26)]
        for sb in sector_bytes:
            sector = dict()
            sector['floor_height'], = unpack('h', sb[0: 2])
            sector['ceiling_height'], = unpack('h', sb[2: 4])
            sector['floor_flat'] = decode_doomstring(sb[4: 12])  # flat name, from the directory
            sector['ceiling_flat'] = decode_doomstring(sb[12: 20])  # flat name, from the directory
            sector['lightlevel'], = unpack('h', sb[20:22])
            sector['special_sector'], = unpack('h', sb[22:24])
            sector['tag'], = unpack('h', sb[24:26]) # tag corresponding to LINEDEF trigger, effect determined by sepcial_sector
            self.append(sector)
        return self

    def add_sector(self, floor_height, ceiling_height, floor_flat, ceiling_flat, lightlevel, special_sector, tag):
        """
        Adds a new sector and returns its id
        :param floor_height:
        :param ceiling_height:
        :param floor_flat:
        :param ceiling_flat:
        :param lightlevel:
        :param special_sector:
        :param tag:
        :return:
        """
        assert floor_height < ceiling_height, "Floor height must be less than ceiling height!"
        self.append({'floor_height':floor_height, 'ceiling_height':ceiling_height, 'floor_flat':floor_flat, 'ceiling_flat':ceiling_flat, 'lightlevel':lightlevel, 'special_sector':special_sector, 'tag':tag})
        return len(self)-1
    def to_bytes(self):
        lump_bytes = bytearray()
        for sector in self:
            lump_bytes += sector['floor_height'].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += sector['ceiling_height'].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += encode_doomstring(sector['floor_flat'])
            lump_bytes += encode_doomstring(sector['ceiling_flat'])
            lump_bytes += sector['lightlevel'].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += sector['special_sector'].to_bytes(2, byteorder='little', signed=True)
            lump_bytes += sector['tag'].to_bytes(2, byteorder='little', signed=True)
        return lump_bytes


class Reject(bytes):
    """ Sector x Sector matrix which tells if an enemy on sector """
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # This seems to be an optional lump so for now it's not implemented
        self = byte_stream
        return self

class Blockmap(dict):
    """
    Precomputed collisions
    """
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # The BLOCKMAP is composed of three parts: the header, the offsets, and the blocklists.
        header = dict()
        header['x_origin'], = unpack('h', byte_stream[0: 2])
        header['y_origin'], = unpack('h', byte_stream[2: 4])
        header['n_cols'], = unpack('h', byte_stream[4: 6])
        header['n_rows'], = unpack('h', byte_stream[6: 8])
        self['header'] = header

        n_blocks = header['n_cols']*header['n_rows']
        self['offsets'] = []
        self['blocklists'] = []

        for i_record in range(n_blocks):
            offset, = unpack('H', byte_stream[8+i_record*2: 8+i_record*2+2])
            self['offsets'].append(offset)
        # Splitting the blocklist byte sequences
        offset_in_bytes = [2*off for off in self['offsets']]
        blocklist_bytes = list()
        if (len(offset_in_bytes)) > 1:
            for start_off, end_off in zip(offset_in_bytes, offset_in_bytes[1:]):
                blocklist_bytes.append(byte_stream[start_off:end_off])
        # Last blocklist
        blocklist_bytes.append(byte_stream[offset_in_bytes[-1]:])

        # Decoding blocklists
        for bb in blocklist_bytes:
            # this row unpacks every integer contained into the blocklist in a list of signed shorts,
            # discarding the first and last int because they are delimiters, respectively 0000 and FFFF
            self['blocklists'].append(list(unpack('h' * (len(bb) // 2), bb)))
        return self

        ### Some tools also build BLOCKMAP, so no writing functions are needed for this lump


def decode_doomstring(byte_string):
    """
    Returns a string from the Doom String format (8 byte - ascii encoded - Null padded strings)
    This function also clean malformed strings (not null-padded or with invalid bytes)
    :param byte_string: a bytes() object
    :return: String
    """
    s = list()
    import sys
    if len(byte_string) > 0:
        for b in byte_string:
            if b == 0:
                break
            try:
                b = (b).to_bytes(1, 'little').decode('ascii')
            except Exception:
                # Encountered an invalid character, just ignore it
                continue
            s.append(b)
        return ''.join(s)
    else:
        return ''

def encode_doomstring(ascii_string):
    if len(ascii_string)<8:
        padding = ['\x00' for i in range(8-len(ascii_string))]
        ascii_string = ascii_string+''.join(padding)
    return bytes(ascii_string[0:8], encoding='ascii')


known_lumps_classes = {'THINGS': Things, 'LINEDEFS': Linedefs, 'SIDEDEFS': Sidedefs, 'VERTEXES': Vertexes, 'SEGS': Segs,
'SSECTORS': SSectors, 'NODES':Nodes, 'SECTORS':Sectors, 'REJECT':Reject, 'BLOCKMAP':Blockmap}
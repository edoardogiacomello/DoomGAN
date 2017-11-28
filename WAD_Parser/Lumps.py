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

class Linedefs(list):
    def __init__(self):
        super()
    def from_bytes(self, byte_stream):
        # Each linedef is 14 bytes long
        linedef_bytes = [byte_stream[s:s + 14] for s in range(0, len(byte_stream), 14)]
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
            sidedef['upper_texture'] = sb[4:12].decode('ascii')  # '-' for simple sidedefs
            sidedef['lower_texture'] = sb[12:20].decode('ascii')  # '-' for simple sidedefs
            sidedef['middle_texture'] = sb[20:28].decode('ascii') # normal or full texture, '-' if transparent
            sidedef['sector'], = unpack('h',sb[28:30]) # Sector the sidedef is facing
            self.append(sidedef)
        # Texture names are from TEXTURE1/2 resources. Wall patches into the directory are referenced through the PNAMES lump
        return self

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
            sector['floor_flat'] = sb[4: 12].decode('ascii')  # flat name, from the directory
            sector['ceiling_flat'] = sb[12: 20].decode('ascii')  # flat name, from the directory
            sector['lightlevel'], = unpack('h', sb[20:22])
            sector['special_sector'], = unpack('h', sb[22:24])
            sector['tag'], = unpack('h', sb[24:26]) # tag corresponding to LINEDEF trigger, effect determined by sepcial_sector
            self.append(sector)
        
        return self

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


known_lumps_classes = {'THINGS': Things, 'LINEDEFS': Linedefs, 'SIDEDEFS': Sidedefs, 'VERTEXES': Vertexes, 'SEGS': Segs,
'SSECTORS': SSectors, 'NODES':Nodes, 'SECTORS':Sectors, 'REJECT':Reject, 'BLOCKMAP':Blockmap}
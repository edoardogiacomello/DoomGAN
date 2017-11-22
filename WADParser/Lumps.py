known_lumps = ['THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS',
'SSECTORS', 'NODES', 'SECTORS', 'REJECT', 'BLOCKMAP']

class Lump(dict):
    #  Only the lumps needed to describe the level and it's "things" are considered
    def __init__(self):
        super()


class Things(dict):
    def __init__(self):
        super()
        self["x"]=None
        self["y"]=None
        self["angle"]=None
        self["type"]=None
        self["options"]=None
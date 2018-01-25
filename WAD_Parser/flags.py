from struct import unpack

def linedef_flags_to_int(impassable = False, block_monsters=False,
                         twosided=False, upper_unpegged=False,
                         lower_unpegged=False, secret=False,
                         block_sound=False, not_on_map=False,
                         already_on_map=False):
    """
    From the Unofficial Doom Specs (http://www.gamers.org/dhs/helpdocs/dmsp1666.html)
    bit     Condition
    0       Impassible
    1       Block Monsters
    2       Two-sided
    3       Upper Unpegged
    4       Lower Unpegged
    5       Secret
    6       Block Sound
    7       Not on Map
    8       Already on Map
    9-15    unused
    :return: The integer representation of the flags
    """
    st = "0000000{:b}{:b}{:b}{:b}{:b}{:b}{:b}{:b}{:b}".format(already_on_map,not_on_map,block_sound,secret,lower_unpegged,upper_unpegged,twosided,block_monsters, impassable)
    return int(st,2)

def int_to_linedef_flags(int_flag):
    st = bin(int_flag).replace("0b", "0"*15)[-16:]
    dict = {
            'already_on_map' : bool(int(st[-9])),
            'not_on_map' : bool(int(st[-8])),
            'block_sound' : bool(int(st[-7])),
            'secret' : bool(int(st[-6])),
            'lower_unpegged' : bool(int(st[-5])),
            'upper_unpegged' : bool(int(st[-4])),
            'twosided' : bool(int(st[-3])),
            'block_monsters' : bool(int(st[-2])),
            'impassable' : bool(int(st[-1]))
            }
    return dict


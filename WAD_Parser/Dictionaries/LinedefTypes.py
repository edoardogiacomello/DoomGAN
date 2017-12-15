# from http://www.gamers.org/dhs/helpdocs/dmsp1666.html
linetypes = {}

linetypes['special'] = {48:0  # Scrolling wall
                        }
linetypes['local_doors'] = {
      1: 10,  # open/close
     26: 12,  # open/close BLUE KEY
     28: 14,  # open/close RED KEY
     27: 16,  # open/close YELLOW KEY
     31: 10,  # open
     32: 12,  # open BLUE KEY
     33: 14,  # open RED KEY
     34: 16,  # open YELLOW KEY
     46: 10,  # open
    117: 10,  # open/close
    118: 10   # open
}
linetypes['remote_doors'] = {
      4: 32, # open,close
     29: 32, # open,close
     90: 32, # open,close
     63: 32, # open,close
      2: 32, # open
    103: 32, # open
     86: 32, # open
     61: 32, # open
      3: 32, # close
     50: 32, # close
     75: 32, # close
     42: 32, # close
     16: 32, # close, then opens
     76: 32, # close, then opens
    108: 32, # open,close
    111: 32, # open,close
    105: 32, # open,close
    114: 32, # open,close
    109: 32, # open
    112: 32, # open
    106: 32, # open
    115: 32, # open
    110: 32, # close
    113: 32, # close
    107: 32, # close
    116: 32, # close
    133: 32, # open BLUE KEY
     99: 32, # open BLUE KEY
    135: 32, # open RED KEY
    134: 32, # open RED KEY
    137: 32, # open YELLOW KEY
    136: 32  # open YELLOW KEY
}

linetypes['ceilings'] = {
     40: 0, # up to HEC
     41: 0, # down to floor
     43: 0, # down to floor
     44: 0, # down to floor + 8
     49: 0, # down to floor + 8
     72: 0  # down to floor + 8

}

linetypes['lifts'] = {
         10: 64, # lift
         21: 64, # lift
         88: 64, # lift
         62: 64, # lift
        121: 64, # lift
        122: 64, # lift
        120: 64, # lift
        123: 64  # lift
}

linetypes['floors'] = {
    119: 0, # up to nhEF
    128: 0, # up to nhEF
     18: 0, # up to nhEF
     69: 0, # up to nhEF
     22: 0, # up to nhEF
     95: 0, # up to nhEF
     20: 0, # up to nhEF
     68: 0, # up to nhEF
     47: 0, # up to nhEF
      5: 0, # up to LIC
     91: 0, # up to LIC
    101: 0, # up to LIC
     64: 0, # up to LIC
     24: 0, # up to LIC
    130: 0, # up to nhEF
    131: 0, # up to nhEF
    129: 0, # up to nhEF
    132: 0, # up to nhEF
     56: 0, # up to LIC - 8, CRUSH
     94: 0, # up to LIC - 8, CRUSH
     55: 0, # up to LIC - 8, CRUSH
     65: 0, # up to LIC - 8, CRUSH
     58: 0, # up 24
     92: 0, # up 24
     15: 0, # up 24
     66: 0, # up 24
     59: 0, # up 24
     93: 0, # up 24
     14: 0, # up 32
     67: 0, # up 32
    140: 0, # up 512
     30: 0, # up ShortestLowerTexture
     96: 0, # up ShortestLowerTexture
     38: 0, # down to LEF
     23: 0, # down to LEF
     82: 0, # down to LEF
     60: 0, # down to LEF
     37: 0, # down to LEF
     84: 0, # down to LEF
     19: 0, # down to HEF
    102: 0, # down to HEF
     83: 0, # down to HEF
     45: 0, # down to HEF
     36: 0, # down to HEF + 8
     71: 0, # down to HEF + 8
     98: 0, # down to HEF + 8
     70: 0, # down to HEF + 8
      9: 0  # donut (see note 12 above)
}

linetypes['stairs'] = {
      8: 64, # stairs
      7: 64, # stairs
    100: 64, # stairs (each up 16 not 8) + crush
    127: 64  # stairs (each up 16 not 8) + crush
}

linetypes['moving_floors'] = {
     53: 64, #  start moving floor
     54: 64, #  stop moving floor
     87: 64, #  start moving floor
     89: 64  #  stop moving floor
}

linetypes['crushing_ceilings'] = {
      6: 0, #  start crushing, fast hurt
     25: 0, #  start crushing, slow hurt
     73: 0, #  start crushing, slow hurt
     77: 0, #  start crushing, fast hurt
     57: 0, #  stop crush
     74: 0, #  stop crush
    141: 0  #  start crushing, slow hurt "Silent"
}

linetypes['exit_level'] = {
     11: 255, #  End level, go to next level
     51: 255, #  End level, go to secret level
     52: 255, #  End level, go to next level
    124: 255  #  End level, go to secret level
}

linetypes['teleport'] = {
     39:192, #  Teleport
     97:192, #  Teleport
    125:192, #  Teleport monsters only
    126:192  #  Teleport monsters only
}

linetypes['light'] = {
     35:0, #  0
    104:0, #  LE (light level)
     12:0, #  HE (light level)
     13:0, #  255
     79:0, #  0
     80:0, #  HE (light level)
     81:0, #  255
     17:0, #  Light blinks (see {4-9-1] type 3)
    138:0, #  255
    139:0  #  0
}

all_linetypes = {} # Contains the list of linedef descriptors, ordered by category
for cat in linetypes:
    for ld_type in linetypes[cat]:
        all_linetypes[ld_type] = linetypes[cat][ld_type]

def get_index_from_type(linedef_type):
    if linedef_type in all_linetypes:
        return all_linetypes[linedef_type]
    return 0

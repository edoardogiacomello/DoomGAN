
linetypes = {}

linetypes['special'] = [48  # Scrolling wall
                       ]
linetypes['local_doors'] = [
      1,  # open/close
     26,  # open/close BLUE KEY
     28,  # open/close RED KEY
     27,  # open/close YELLOW KEY
     31,  # open
     32,  # open BLUE KEY
     33,  # open RED KEY
     34,  # open YELLOW KEY
     46,  # open
    117,  # open/close
    118  # open
]
linetypes['remote_doors'] = [
  4, # open,close
 29, # open,close
 90, # open,close
 63, # open,close
  2, # open
103, # open
 86, # open
 61, # open
  3, # close
 50, # close
 75, # close
 42, # close
 16, # close, then opens
 76, # close, then opens
108, # open,close
111, # open,close
105, # open,close
114, # open,close
109, # open
112, # open
106, # open
115, # open
110, # close
113, # close
107, # close
116, # close
133, # open BLUE KEY
 99, # open BLUE KEY
135, # open RED KEY
134, # open RED KEY
137, # open YELLOW KEY
136 # open YELLOW KEY
]
linetypes['ceilings'] = [

 40, # up to HEC
 41, # down to floor
 43, # down to floor
 44, # down to floor + 8
 49, # down to floor + 8
 72 # down to floor + 8

]
linetypes['lifts'] = [

 10, # lift
 21, # lift
 88, # lift
 62, # lift
121, # lift
122, # lift
120, # lift
123 # lift

]
linetypes['floors'] = [

119, # up to nhEF
128, # up to nhEF
 18, # up to nhEF
 69, # up to nhEF
 22, # up to nhEF
 95, # up to nhEF
 20, # up to nhEF
 68, # up to nhEF
 47, # up to nhEF
  5, # up to LIC
 91, # up to LIC
101, # up to LIC
 64, # up to LIC
 24, # up to LIC
130, # up to nhEF
131, # up to nhEF
129, # up to nhEF
132, # up to nhEF
 56, # up to LIC - 8, CRUSH
 94, # up to LIC - 8, CRUSH
 55, # up to LIC - 8, CRUSH
 65, # up to LIC - 8, CRUSH
 58, # up 24
 92, # up 24
 15, # up 24
 66, # up 24
 59, # up 24
 93, # up 24
 14, # up 32
 67, # up 32
140, # up 512
 30, # up ShortestLowerTexture
 96, # up ShortestLowerTexture
 38, # down to LEF
 23, # down to LEF
 82, # down to LEF
 60, # down to LEF
 37, # down to LEF
 84, # down to LEF
 19, # down to HEF
102, # down to HEF
 83, # down to HEF
 45, # down to HEF
 36, # down to HEF + 8
 71, # down to HEF + 8
 98, # down to HEF + 8
 70, # down to HEF + 8
  9, # donut (see note 12 above)

]
linetypes['stairs'] = [

  8, # stairs
  7, # stairs
100, # stairs (each up 16 not 8) + crush
127 # stairs (each up 16 not 8) + crush

]
linetypes['moving_floors'] = [

 53, #  start moving floor
 54, #  stop moving floor
 87, #  start moving floor
 89 #  stop moving floor

]
linetypes['crushing_ceilings'] = [

  6, #  start crushing, fast hurt
 25, #  start crushing, slow hurt
 73, #  start crushing, slow hurt
 77, #  start crushing, fast hurt
 57, #  stop crush
 74, #  stop crush
141 #  start crushing, slow hurt "Silent"

]
linetypes['exit_level'] = [

 11, #  End level, go to next level
 51, #  End level, go to secret level
 52, #  End level, go to next level
124 #  End level, go to secret level

]
linetypes['teleport'] = [

 39, #  Teleport
 97, #  Teleport
125, #  Teleport monsters only
126 #  Teleport monsters only

]
linetypes['light'] = [

 35, #  0
104, #  LE (light level)
 12, #  HE (light level)
 13, #  255
 79, #  0
 80, #  HE (light level)
 81, #  255
 17, #  Light blinks (see [4-9-1] type 3)
138, #  255
139 #  0

]

all_linetypes = list() # Contains the list of linedef descriptors, ordered by category

for category in ['special', 'local_doors', 'remote_doors', 'ceilings',  'monsters', 'ammunitions', 'weapons', 'powerups', 'artifacts']:
    all_linetypes += linetypes[category]

# ENCODING
# 0: NO TRIGGER
# ?: MANUAL DOOR
#
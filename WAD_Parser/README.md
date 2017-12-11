**WAD_PARSER readme**
=======
Python utility for Doom WAD manipulation and feature extraction by Edoardo Giacomello <edoardo.giacomello1990@gmail.com>
***

Main features
 -----------
* Reading of WAD files as structured python dictionaries (Only structural features are decoded)
* Scalar feature extraction from WAD files
* Different kinds of level mapping:
    * Walls only (1-bit greyscale)
    * Heightmaps
    * "Floor" coloring (connected parts are enumerated by color)
    * "Things" map
* Simplified creation of new levels via vertex specification
* Image - to - WAD conversion
* Good fault tolerance: Skips most common bug found in some editors, leading to malformed WAD files. 

Usage
 -----------

### Reading a file
```
from WAD_Parser.WADReader import WADReader   
reader = WADReader()
wad = reader.read("path/to/file.wad")["wad"]
```

### Extrating features and maps
```
wad_with_features = reader.extract("path/to/file.wad")
wad = wad_with_features["wad"]
levels = wad["levels"]
features = level["features"]
maps = level["maps"]
```
### Creating a level from scratch
```
from WAD_Parser.WADReader import WADWriter 
# Let's create a new WAD
writer = WADWriter()
# Declare a level
mine.add_level('MAP01')
# Create a big sector, by specifying its vertices (in clockwise order)
mine.add_sector([(1000,1000),(1000,-1000), (-1000,-1000), (-1000,1000) ])
# set the starting position for the player 1
mine.set_start(0,0)
# Let's add a Cacodemon to make things more interesting
mine.add_thing(x=500,y=500,thing_type=3005, options=7) 
# Save the wad file. "bsp" command should work in your shell for this.
wad_mine = mine.save('path/to/file.wad')
```


# Stub for converting Tile representation to png.

from PIL import Image
import os
folder = './dataset/CommunityLevels/WADs/Doom/Processed/'
tiles = {
        "-" : ["empty","out of bounds"],
        "X" : ["solid","wall"],
        "." : ["floor","walkable"],
        "," : ["floor","walkable","stairs"],
        "E" : ["enemy","walkable"],
        "W" : ["weapon","walkable"],
        "A" : ["ammo","walkable"],
        "H" : ["health","armor","walkable"],
        "B" : ["explosive barrel","walkable"],
        "K" : ["key","walkable"],
        "<" : ["start","walkable"],
        "T" : ["teleport","walkable","destination"],
        ":" : ["decorative","walkable"],
        "L" : ["door","locked"],
        "t" : ["teleport","source","activatable"],
        "+" : ["door","walkable","activatable"],
        ">" : ["exit","activatable"] }
tile_list = list(tiles.keys())
for tile in tile_list:
    col = int(255/19*list(tiles.keys()).index(tile))
    tiles[tile] = col

for file in os.listdir(folder):
    with open(folder+file, 'r') as imgfile:
        lines = [line.strip() for line in  imgfile.readlines()]
        n_lines = len(lines)
        n_cols = len(lines[0])

        img = Image.new('RGB', (n_lines, n_cols), "black")  # create a new black image
        pixels = img.load()  # create the pixel map

        ip = 0 # pixel counters
        jp = 0
        for i in lines:  # for every tile:
            for j in i:
                c = tiles[j]
                img.putpixel((ip,jp), c)
                jp+=1
            jp=0
            ip+=1

        #img.show()




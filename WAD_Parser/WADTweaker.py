from DoomDataset import DoomDataset
from WAD_Parser.WADEditor import WADWriter
from WAD_Parser.WADEditor import WADReader

class WADTweaker():
    def __init__(self):
        """
        Loads the dataset and converts each level back to a .wad file, then compares the feature extracted from the
        original wad with those of the reconstructed wad, allowing to tweak the filters in order to have the minimum
        reconstruction error.
        """

    def build_levels(self, max=32):

        # Create a new WAD
        writer = WADWriter()

        for index in range(max):
            print("Building level {}".format(index))
            pattern = "./../artifacts/generated_samples/sample{}_map_{}_generated.png"
            floormap = pattern.format(index, "floormap")
            heightmap = pattern.format(index, "heightmap")
            wallmap = pattern.format(index, "wallmap")
            thingsmap = pattern.format(index, "thingsmap")
            writer.add_level(name='MAP{i:02d}'.format(i=index+1))
            writer.from_images_v2(floormap, heightmap, wallmap, thingsmap)
        writer.save('./../artifacts/DOOMPCGML.wad')

    def build_test_level(self):
        # Let's create a new WAD
        writer = WADWriter()
        # Declare a level
        writer.add_level('MAP01')
        # Create a big sector, by specifying its vertices (in clockwise order)
        big_room = writer.add_sector([(1000, 1000), (1000, -1000), (-1000, -1000), (-1000, 1000)])
        # Create a "door". It must be specified conuter-clockwise
        door = writer.add_door([(100, 100), (-100, 100), (-100, -100), (100, -100)], remote=True, parent_sector=big_room)
        # Create a switch
        writer.add_trigger([(-150+32, -150+32), (-150-32, -150+32), (-150-32, -150-32), (-150+32, -150-32)], parent_sector=big_room, trigger_type=63, trigger_tag=door)
        # Create a small sector with a different height
        small_step = writer.add_sector(list(reversed([(700+32, 700+32), (700-32, 700+32), (700-32, 700-32), (700+32, 700-32)])),
                                       floor_height=32,
                                       kw_sidedef={'upper_texture':'BRONZE1', 'lower_texture':'BRONZE1', 'middle_texture':'-'},
                                       kw_linedef={'type':0, 'trigger':0, 'flags':4},
                                       surrounding_sector_id=big_room)
        # set the starting position for the player 1
        writer.set_start(-700, -700)
        # Let's add a Cacodemon to make things more interesting
        #writer.add_thing(x=500, y=500, thing_type=3005, options=7)
        # Save the wad file. "bsp" command should work in your shell for this to work.
        wad_mine = writer.save('./../artifacts/test.wad')


    def inspect_doom2(self):
        reader = WADReader()
        wad = reader.extract('./Doom2.wad')
        level = wad['levels'][0]
        maps = level['maps']
        writer = WADWriter()
        writer.add_level('MAP01')
        writer.from_images(heightmap=maps['heightmap'], floormap=maps['floormap'], wallmap=maps['wallmap'], thingsmap=None, debug=False)
        writer.save('./test.wad')
        pass
if __name__ == '__main__':
    WADTweaker().build_levels()
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

    def test_reconstruction(self, database_json):
        # Open the database
        dataset = DoomDataset()
        levels = dataset.read_from_json(database_json)
        for l in levels:
            # Fetch a feature map
            floormap = dataset.get_path_of(l['path_floormap'])
            wallmap = dataset.get_path_of(l['path_wallmap'])
            thingsmap = dataset.get_path_of(l['path_thingsmap'])
            # Create a new WAD
            writer = WADWriter()
            writer.from_images(floormap, wallmap, thingsmap, debug=True)
            writer.save('/home/edoardo/Desktop/doom/test.wad')
            reader = WADReader()
            reconstructed = reader.extract('/home/edoardo/Desktop/doom/test.wad')['levels'][0]


    def build_level(self, index):

        floormap = '/home/edoardo/Projects/DoomPCGML/DoomLevelsGAN/generated_samples/level{}_map_floormap.png'.format(index)
        wallmap = '/home/edoardo/Projects/DoomPCGML/DoomLevelsGAN/generated_samples/level{}_map_wallmap.png'.format(index)
        thingsmap = '/home/edoardo/Projects/DoomPCGML/DoomLevelsGAN/generated_samples/level{}_map_thingsmap.png'.format(index)

        # Create a new WAD
        writer = WADWriter()
        writer.from_images(floormap, wallmap, thingsmap, debug=True)
        writer.save('/home/edoardo/Desktop/doom/test.wad')



if __name__ == '__main__':
    #WADTweaker().test_reconstruction('/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json')
    WADTweaker().build_level(1)
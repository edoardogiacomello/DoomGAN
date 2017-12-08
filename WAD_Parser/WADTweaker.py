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
            img_map = dataset.get_path_of(l['path_floormap'])
            # Create a new WAD
            writer = WADWriter()
            writer.from_floormap(img_map)
            writer.set_start(2500,2500)
            writer.save('/home/edoardo/Desktop/doom/test.wad')
            reader = WADReader()
            reconstructed = reader.extract('/home/edoardo/Desktop/doom/test.wad')['levels'][0]



            break


if __name__ == '__main__':
    tweaker = WADTweaker().test_reconstruction("/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json")
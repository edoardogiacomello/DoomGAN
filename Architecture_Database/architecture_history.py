# THIS FILE CONTAINS THE ARCHITECTURES THAT HAVE BEEN TESTED DURING THE TRAINING PROCESS
# 28/12/2017
features = ['height', 'width',
           'number_of_sectors',
           'sector_area_avg',
           'sector_aspect_ratio_avg',
           'lines_per_sector_avg',
           'walkable_percentage',
           'level_extent',
           'level_solidity',
           'artifacts_per_walkable_area',
           'powerups_per_walkable_area',
           'weapons_per_walkable_area',
           'ammunitions_per_walkable_area',
           'keys_per_walkable_area',
           'monsters_per_walkable_area',
           'obstacles_per_walkable_area',
           'decorations_per_walkable_area']
# Define here which maps to use (you can find a list in /WAD_Parser/Dictionaries/Features.py )
maps = ['floormap', 'heightmap', 'wallmap', 'thingsmap', 'triggermap']

# Generator network layers
g_layers = [
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 16, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 8, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 4, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 2, 'remove_artifacts': False},
]
# Discriminator network layers
d_layers = [
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 2, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 4, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 8, 'remove_artifacts': False},
    {'stride': (2, 2), 'kernel_size': (4, 4), 'n_filters': 64 * 16, 'remove_artifacts': False},
]

###
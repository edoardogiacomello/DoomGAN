# This file defines the the network layers and the input features.
# Define here which features to use (you can find a list in /WAD_Parser/Dictionaries/Features.py )
features = ['level_equivalent_diameter',
            'level_major_axis_length',
            'level_minor_axis_length',
            'level_solidity',
            'nodes',
            'distmap-skew',
            'distmap-kurt']

# SEED OF THE NETWORK
seed = 123456789
# Define here which maps to use (you can find a list in /WAD_Parser/Dictionaries/Features.py )
maps = ['floormap', 'heightmap', 'wallmap', 'thingsmap']

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
import DoomDataset
dd = DoomDataset.DoomDataset()



constraints = [
    lambda x: x['width'] <= 32*128,  lambda x: x['height'] <= 32*128,
    lambda x: abs(x['floor_height_avg']) <= 352,
    lambda x: abs(x['lines_per_sector_avg']) <= 200,
    lambda x: abs(x['floor_height_max']) <= 2000,
    lambda x: abs(x['floor_height_min']) <= 2000,
    #lambda x: abs(x['floors']) == 1
]
features_to_use = ['height', 'width',
                   'number_of_sectors',
                   'sector_area_avg',
                   'sector_aspect_ratio_avg',
                   'lines_per_sector_avg',
                   'floor_height_avg',
                   'floor_height_max',
                   'floor_height_min',
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

#dd.plot_joint_feature_distributions('/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json', features_to_use,
#                                    constraints_lambdas=constraints).savefig('./../dataset/statistics/128_one_floor')
dd.to_TFRecords(json_db='/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json',
                output_path='/run/media/edoardo/BACKUP/Datasets/DoomDataset/128-many-floors',
                validation_size=0.3,
                target_size=(128,128),
                constraints_lambdas=constraints)

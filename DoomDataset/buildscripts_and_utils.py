import DoomDataset
dd = DoomDataset.DoomDataset()



constraints = [
    lambda x: x['width'] < 32*256,
    lambda x: x['height'] < 32*256,
    lambda x: abs(x['floor_height_avg']) <= 352,
    lambda x: abs(x['lines_per_sector_avg']) <= 200,
    lambda x: abs(x['floor_height_max']) <= 2000,
    lambda x: abs(x['floor_height_min']) <= 2000,
    lambda x: abs(x['floors']) == 1
]

#dd.plot_joint_feature_distributions('', features_to_use,
#                                    constraints_lambdas=constraints).savefig('./../dataset/statistics/128_one_floor')
dd.to_TFRecords(json_db='',
                output_path='',
                validation_size=0.3,
                target_size=(256,256),
                constraints_lambdas=constraints)

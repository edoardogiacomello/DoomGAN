import DoomDataset as dd
d = dd.DoomDataset()
d.recompute_features(root='/run/media/edoardo/BACKUP/Datasets/DoomDataset/',
                     old_json_db='/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json',
                     new_json_db='/run/media/edoardo/BACKUP/Datasets/DoomDataset/dataset.json.new')
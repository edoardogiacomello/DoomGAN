import  numpy as np
import glob
import skimage.io as io
import DoomLevelsGAN.inception.inception_score as inception
import itertools
import os
import tensorflow as tf

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def get_inception_score():

    #files = glob.glob('/home/edoardo/Projects/DoomPCGML/DoomLevelsGAN/generated_samples/*.png')
    dataset = glob.glob('/run/media/edoardo/BACKUP/Datasets/DoomDataset/Processed/*.png')

    dataset_batched = grouper(32, dataset)
    for fn_batch in dataset_batched:
        batch = list()
        for fn in fn_batch:
            img = io.imread(fn) if os.path.exists(fn) else np.zeros((128,128))
            arr = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
            batch.append(arr)
        print(inception.get_inception_score(batch))




if __name__ == '__main__':
    get_inception_score()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import DoomLevelsGAN.DoomGAN as nn


def plot_feature_distributions_1v1(true_data, gen_data):
    """
    Shows a joint plot for each feature (old version)
    :param true_data:
    :param gen_data:
    :return:
    """
    for f_id, f_name in enumerate(nn.features):
        feats = np.stack([true_data[:, f_id], gen_data[:, f_id]], axis=1).astype(np.float32)
        cols = ['True {}'.format(f_name), 'Generated {}'.format(f_name)]
        pdata = pd.DataFrame(feats, columns=cols)
        g = sb.jointplot(cols[0], cols[1], data=pdata, kind="reg", size=7, space=0)
        print("Showing plot {} of {}".format(f_id, len(nn.features)))
        plt.show()

def generate_results_and_save(samples_for_map):
    """
    Loads the network and generates <samples_for_map> features for every map in the training set.
    Results are saved into the artifacts folder.
    :param samples_for_map: How many samples to generate for each "true" map.
    :return:
    """
    true_features, generated_features, noise = nn.gan.evaluate_samples_distribution(n=samples_for_map)
    np.save(nn.FLAGS.ref_sample_folder+'results_true.npy', true_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_gen.npy', generated_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_noise.npy', noise)

def load_results_from_file():
    """Loads previously saved results from the artifacts folder"""
    true = np.load(nn.FLAGS.ref_sample_folder+'results_true.npy').astype(np.float32)
    gen = np.load(nn.FLAGS.ref_sample_folder+'results_gen.npy')[:,:].astype(np.float32)
    return true, gen


def distribution_visualization_new():
    """ WORK IN PROGRESS"""
    true, gen = load_results_from_file()
    # Pandas dataframe generation (for visualizing in seaborn)
    features = nn.features
    sets = ['True', 'Generated']
    # Expand the arrays for matching the pandas syntax
    columns = [[s for s in sets for f in features], [f for s in sets for f in features]]
    data = np.reshape(np.concatenate([true, gen], axis=-1).flatten(), (true.shape[0], len(features)*len(sets)))
    pdata = pd.DataFrame(data, columns=columns, names=['source', 'features'])




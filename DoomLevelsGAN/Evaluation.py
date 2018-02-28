import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import DoomLevelsGAN.DoomGAN as nn
from WAD_Parser.Dictionaries import Features as all_features


def generate_results_and_save(samples_for_map):
    """
    Loads the network and generates <samples_for_map> features for every map in the training set.
    Results are saved into the artifacts folder.
    :param samples_for_map: How many samples to generate for each "true" map.
    :return:
    """
    names, true_features, oth_true_features, generated_features, oth_gen_features, noise = nn.gan.evaluate_samples_distribution(n=samples_for_map)
    np.save(nn.FLAGS.ref_sample_folder+'results_names.npy', names)
    np.save(nn.FLAGS.ref_sample_folder+'results_true.npy', true_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_true_oth.npy', oth_true_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_gen.npy', generated_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_gen_oth.npy', oth_gen_features)
    np.save(nn.FLAGS.ref_sample_folder+'results_noise.npy', noise)

def load_results_from_file():
    """Loads previously saved results from the artifacts folder"""
    true = np.load(nn.FLAGS.ref_sample_folder+'results_true.npy').astype(np.float32)
    gen = np.load(nn.FLAGS.ref_sample_folder+'results_gen.npy')[:,:].astype(np.float32)
    return true, gen

def clean_nans(a):
    """
    Removes rows from a that contains non numerical values
    :param a:
    :return:
    """
    return a[~np.isnan(a).any(axis=1)]

def distribution_visualization_1v1(feature_names):
    """ WORK IN PROGRESS"""
    true, gen = load_results_from_file()
    # fixing the first dimension
    gen = np.mean(gen, axis=-1)
    tc = clean_nans(true)
    gc = clean_nans(gen)
    features = nn.features
    # Showing True features distribution vs Generated
    for f, fname in enumerate(features):
        axt = sb.kdeplot(tc[:,f], label="True")
        tmean = tc[:, f].mean()
        sb.rugplot([tmean], height=1, ax=axt, ls="--", color=axt.get_lines()[-1].get_color())

        axg = sb.kdeplot(gc[:,f], label="Generated")
        gmean = gc[:, f].mean()
        sb.rugplot([gmean], height=1, ax=axg, ls="--", color=axg.get_lines()[-1].get_color())

        axt.set_xlabel("{}".format(fname))
        axt.figure.canvas.set_window_title("1v1_{}".format(fname))
        plt.show()



# distribution_visualization_new()
input_features = nn.features
oth_features = [f for f in all_features.features_for_evaluation if f not in input_features]

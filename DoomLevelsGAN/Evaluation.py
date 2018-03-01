import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import DoomLevelsGAN.DoomGAN as nn
from WAD_Parser.Dictionaries import Features as all_features
import os


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

def load_results_from_files():
    """Loads previously saved results from the artifacts folder"""
    names = np.load(nn.FLAGS.ref_sample_folder + 'results_names.npy')
    true_features = np.load(nn.FLAGS.ref_sample_folder + 'results_true.npy').astype(np.float32)
    oth_true_features = np.load(nn.FLAGS.ref_sample_folder + 'results_true_oth.npy').astype(np.float32)
    generated_features = np.load(nn.FLAGS.ref_sample_folder + 'results_gen.npy').astype(np.float32)
    oth_gen_features = np.load(nn.FLAGS.ref_sample_folder + 'results_gen_oth.npy').astype(np.float32)
    noise = np.load(nn.FLAGS.ref_sample_folder + 'results_noise.npy').astype(np.float32)
    return names, true_features, oth_true_features, generated_features, oth_gen_features, noise

def clean_nans(a):
    """
    Removes rows from a that contains non numerical values
    :param a:
    :return:
    """
    if len(a.shape) == 1:
        return a[~np.isnan(a)]
    return a[~np.isnan(a).any(axis=1)]

def distribution_visualization_1v1(colors={'True':'red', 'Gen':'dodgerblue'}):
    """
    Plots true vs generated distribution (1 generated vector for each true one) for each feature that is possible to extract from generated samples (both in input to the network or not)
    Requires data file 'results_*.npy' to be in the artifacts folder.
    Saves the image results in the artifact folder
    :param colors:
    :return:
    """
    features_output_folder = nn.FLAGS.ref_sample_folder + "graphs/1v1/input_features/"
    oth_features_output_folder = nn.FLAGS.ref_sample_folder + "graphs/1v1/other_features/"

    os.makedirs(features_output_folder,exist_ok=True)
    os.makedirs(oth_features_output_folder,exist_ok=True)

    oth_features = [f for f in all_features.features_for_evaluation if f not in nn.features]

    names, true, oth_true, gen, oth_gen, noise = load_results_from_files()
    # fixing the first dimension
    assert gen.shape[-1] == 1, "The loaded generated results have {} samples per map instead of 1. Make sure of loading the correct result set".format(gen.shape[-1])
    gen = np.squeeze(gen,-1)
    oth_gen = np.squeeze(oth_gen,-1)



    # Showing True features distribution vs Generated for the input features
    for f, fname in enumerate(nn.features):
        # Clearing rows containing NaNs, from now on the correspondence between indices/level name is lost

        tc = clean_nans(true[:,f])
        gc = clean_nans(gen[:,f])

        fig = plt.figure()
        axt = sb.rugplot([tc.mean()], height=1, ls="--", color=colors['True'], linewidth=0.75)
        sb.kdeplot(tc, ax=axt, label="True",ls="--", color=colors['True'])

        axg = sb.rugplot([gc.mean()], height=1, color=colors['Gen'], linewidth=0.75)
        sb.kdeplot(gc, ax=axg, label="Generated", color=colors['Gen'])

        axt.set_xlabel("{}".format(fname))
        fig_name = "1v1_{}".format(fname)
        axt.figure.canvas.set_window_title(fig_name)

        fig.savefig(features_output_folder+'/png/'+fig_name+'.png')
        fig.savefig(features_output_folder+'/pdf/'+fig_name+'.pdf')
        plt.close(fig)

    for f, fname in enumerate(oth_features):
        otc = clean_nans(oth_true[:, f])
        ogc = clean_nans(oth_gen[:, f])
        fig = plt.figure()
        axt = sb.rugplot([otc.mean()], height=1, ls="--", color=colors['True'], linewidth=0.75)
        sb.kdeplot(otc, ax=axt, label="True", ls="--",  color=colors['True']) if np.unique(otc, axis=0).size > 1 else None
        # Don't show the distribution if data contains only a value -> matrix is singular


        axg = sb.rugplot([ogc.mean()], height=1, color=colors['Gen'], linewidth=0.75)
        sb.kdeplot(ogc, ax=axg, label="Generated",  color=colors['Gen']) if np.unique(ogc, axis=0).size > 1 else None
        # Don't show the distribution if data contains only a value -> matrix is singular

        axt.set_xlabel("{}".format(fname))
        fig_name = "1v1_{}".format(fname)
        axt.figure.canvas.set_window_title(fig_name)
        axt.figure.savefig(oth_features_output_folder+'/png/'+fig_name+'.png')
        axt.figure.savefig(oth_features_output_folder+'/pdf/'+fig_name+'.pdf')
        plt.close(fig)



# distribution_visualization_new()
input_features = nn.features

#distribution_visualization_1v1()
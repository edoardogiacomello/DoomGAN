"""
This script works as stand-alone package from the network.
It expects the following files (in the same directory of this script):

- a set of .csv files for 1v1 comparison containing the feature values from the network: /input_files/result_*.csv

- network_architecture.py taken from the network that you are evaluating (root)
- features.py taken from the WAD_Parser (already included)

"""
import Features
import network_architecture as arch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import os
import csv
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ks_2samp

input_features = arch.features
oth_features = [f for f in Features.features_for_evaluation if f not in input_features]

# Folder specifications
in_folder = './input_files/'
out_folder = './output_files/'
out_1v1_folder = out_folder + '1v1/'
features_output_folder = out_folder + "./graphs/1v1/input_features/"
oth_features_output_folder = out_folder + "graphs/1v1/other_features/"



def npy_to_csv():
    """Loads previously saved results from the artifacts folder"""
    names = np.load(in_folder + 'results_names.npy').astype(np.unicode_)
    true_features = np.load(in_folder + 'results_true.npy').astype(np.float32)
    oth_true_features = np.load(in_folder + 'results_true_oth.npy').astype(np.float32)
    generated_features = np.load(in_folder + 'results_gen.npy').astype(np.float32)
    oth_gen_features = np.load(in_folder + 'results_gen_oth.npy').astype(np.float32)
    noise = np.load(in_folder + 'results_noise.npy').astype(np.float32)

    # HEADER GENERATION
    names_header = 'sample_path'
    in_header = ','.join(input_features)
    oth_header = ','.join(oth_features)
    noise_header = ','.join(["z{}".format(i) for i in range(noise.shape[1])])

    np.savetxt(in_folder + 'results_{}.csv'.format("names"), names, fmt="%s", delimiter=',', header=names_header, comments='')
    np.savetxt(in_folder + 'results_{}.csv'.format("true"), true_features, delimiter=',', header=in_header, comments='')
    np.savetxt(in_folder + 'results_{}.csv'.format("true_oth"), oth_true_features, delimiter=',', header=oth_header, comments='')
    np.savetxt(in_folder + 'results_{}.csv'.format("gen"), generated_features, delimiter=',', header=in_header, comments='')
    np.savetxt(in_folder + 'results_{}.csv'.format("gen_oth"), oth_gen_features, delimiter=',', header=oth_header, comments='')
    np.savetxt(in_folder + 'results_{}.csv'.format("noise"), noise, delimiter=',', header=noise_header, comments='')

def load_results_from_files(mode='numpy'):
    """Loads previously saved results from the artifacts folder. If mode is 'numpy' then returns
    the unpacked set of numpy vectors names, true_features, oth_true_features, generated_features, oth_gen_features, noise.
    If mode is 'pandas' then returns a MultiIndex dataset which can be addressed by.
    data[<feature_name>][<true | gen>].
    For example:

    data['nodes']['true'] is the dataset containing the room count of the true levels
    data.xs('true') gives the true dataset
    data.xs('gen') gives the gen dataset
    """
    assert mode in {'numpy', 'pandas'}, "Please specify a mode that is either 'numpy' or 'pandas'"

    if mode == 'numpy':
        names = np.loadtxt(in_folder + 'results_names.csv', delimiter=',', dtype=np.str, skiprows=1)
        true_features = np.loadtxt(in_folder + 'results_true.csv', delimiter=',', dtype=np.float32, skiprows=1)
        oth_true_features = np.loadtxt(in_folder + 'results_true_oth.csv', delimiter=',', dtype=np.float32, skiprows=1)
        generated_features = np.loadtxt(in_folder + 'results_gen.csv', delimiter=',', dtype=np.float32, skiprows=1)
        oth_gen_features = np.loadtxt(in_folder + 'results_gen_oth.csv', delimiter=',', dtype=np.float32, skiprows=1)
        noise = np.loadtxt(in_folder + 'results_noise.csv', delimiter=',', dtype=np.float32, skiprows=1)
        return names, true_features, oth_true_features, generated_features, oth_gen_features, noise
    elif mode == 'pandas':
        if len(input_features) > 0:
            #names = pd.read_csv(in_folder + 'results_names.csv', header=0, comment='#', delimiter=',')
            true_features = pd.read_csv(in_folder + 'results_true.csv', header=0, comment='#', delimiter=',')
            oth_true_features = pd.read_csv(in_folder + 'results_true_oth.csv', header=0, comment='#', delimiter=',')
            generated_features = pd.read_csv(in_folder + 'results_gen.csv', header=0, comment='#', delimiter=',')
            oth_gen_features = pd.read_csv(in_folder + 'results_gen_oth.csv', header=0, comment='#', delimiter=',')
            pd_true, pd_gen = pd.concat([true_features, oth_true_features], axis=1), pd.concat([generated_features, oth_gen_features], axis=1)
            return pd.concat({'true':pd_true, 'gen':pd_gen}, axis=1)
        else:
            oth_true_features = pd.read_csv(in_folder + 'results_true_oth.csv', header=0, comment='#', delimiter=',')
            oth_gen_features = pd.read_csv(in_folder + 'results_gen_oth.csv', header=0, comment='#', delimiter=',')
            return pd.concat({'true': oth_true_features, 'gen': oth_gen_features}, axis=1)


def distribution_visualization_1v1(mode='pdf', colors={'True':'red', 'Gen':'dodgerblue'}):
    """
    Plots true vs generated distribution (1 generated vector for each true one) for each feature that is possible to extract from generated samples (both in input to the network or not)
    Requires data file 'results_*.csv' to be in the input_files folder.
    Saves graphs and numerical results in the output folder
    :param colors:
    :return:
    """
    assert mode in {'pdf', 'cdf'}
    os.makedirs(features_output_folder+"png/",exist_ok=True)
    os.makedirs(features_output_folder+"pdf/",exist_ok=True)
    os.makedirs(oth_features_output_folder+"png/",exist_ok=True)
    os.makedirs(oth_features_output_folder+"pdf/",exist_ok=True)

    stat_test_input = list()


    data = load_results_from_files('pandas')
    # Dropping levels that have nan data
    #data = data.dropna(axis=0, how='any')

    # Showing True features distribution vs Generated for the input features

    for f, fname in enumerate(input_features + oth_features):
        idx = pd.IndexSlice
        # This is the pandas way to select the same column on each level of a multi-index
        sliced_data = data.loc[idx[:], idx[:, fname]]
        clean_data = sliced_data.dropna(axis=0, how='any')

        tc = clean_data['true'][fname]
        gc = clean_data['gen'][fname]

        fig = plt.figure()
        try:
            if mode == 'pdf':
                axt = sb.rugplot([tc.median()], height=1, ls="--", color=colors['True'], linewidth=0.75)
                sb.kdeplot(tc, ax=axt, label="True",ls="--", color=colors['True'])
                axg = sb.rugplot([tc.median()], height=1, color=colors['Gen'], linewidth=0.75)
                sb.kdeplot(gc, ax=axg, label="Generated", color=colors['Gen'])
                pass
            else:
                axt = sb.kdeplot(tc, label="True", ls="--", color=colors['True'], cumulative=True)
                axg = sb.kdeplot(gc, label="Generated", color=colors['Gen'], cumulative=True)

            # STATISTICAL TESTS
            w_stat, w_pvalue = wilcoxon(tc, gc)
            t_stat, t_pvalue = ttest_ind(tc, gc, nan_policy='omit')
            u_stat, u_pvalue = mannwhitneyu(tc, gc, alternative='two-sided')
            ks_stat, ks_pvalue = ks_2samp(tc, gc)
        except:
            print("Failed to plot feature {} because the matrix is singular. This feature may be not informative with the considered dataset".format(fname))
            stat_test_input.append((fname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        # When adding a new stat test make sure of also editing the csv header after the for loop
        stat_test_input.append((fname, w_stat, w_pvalue, t_stat, t_pvalue, u_stat, u_pvalue, ks_stat, ks_pvalue))

        # GRAPH ANNOTATIONS
        #print("{}\t{}\t{}".format(fname, w_pvalue, t_pvalue))
        axt.set_xlabel("{}\nWilcoxon:{} \n T-Test:{} \n U-Test:{} \n KS-Test:{} \n KS-Stat:{}".format(fname, w_pvalue, t_pvalue, u_pvalue, ks_pvalue, ks_stat), fontsize=16)
        fig_name = "1v1_{}".format(fname)
        axt.figure.canvas.set_window_title(fig_name)
        fig.tight_layout()

        # SAVING GRAPHS
        if fname in input_features:
            fig.savefig(features_output_folder+'png/'+fig_name+'.png')
            fig.savefig(features_output_folder+'pdf/'+fig_name+'.pdf')
        else:
            fig.savefig(oth_features_output_folder + 'png/' + fig_name + '.png')
            fig.savefig(oth_features_output_folder + 'pdf/' + fig_name + '.pdf')
        plt.close(fig)

    # WRITING STATS TO CSV
    stat_columns = ['feature', 'W-stat', 'W-pvalue', 'T-stat', 'T-pvalue', 'U-stat', 'U-pvalue', 'KS-stat', 'KS-pvalue']
    with open(out_folder+"/stat_test_result.csv", 'w') as csvout:
        writer = csv.writer(csvout, delimiter=',')
        writer.writerow(stat_columns)
        for stat in stat_test_input:
            writer.writerow(stat)

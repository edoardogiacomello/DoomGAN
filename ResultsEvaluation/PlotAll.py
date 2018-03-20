import Features
import network_architecture as arch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import os
import csv
import glob
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ks_2samp


# PATS SPECIFICATION
input_files = './input_files/'
output_files = './output_files/'

def load_results_from_files(in_folder, mode='numpy'):
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

        # If the network has no input features, the corresponding file is empty
        try:
            true_features = pd.read_csv(in_folder + 'results_true.csv', header=0, comment='#', delimiter=',')
            generated_features = pd.read_csv(in_folder + 'results_gen.csv', header=0, comment='#', delimiter=',')
        except pd.errors.EmptyDataError:
            oth_true_features = pd.read_csv(in_folder + 'results_true_oth.csv', header=0, comment='#', delimiter=',')
            oth_gen_features = pd.read_csv(in_folder + 'results_gen_oth.csv', header=0, comment='#', delimiter=',')
            return pd.concat({'true': oth_true_features, 'gen': oth_gen_features}, axis=1)
        oth_true_features = pd.read_csv(in_folder + 'results_true_oth.csv', header=0, comment='#', delimiter=',')
        oth_gen_features = pd.read_csv(in_folder + 'results_gen_oth.csv', header=0, comment='#', delimiter=',')
        pd_true, pd_gen = pd.concat([true_features, oth_true_features], axis=1), pd.concat([generated_features, oth_gen_features], axis=1)
        return pd.concat({'true':pd_true, 'gen':pd_gen}, axis=1)


def plot_all(cumulative=True):
    """
    Plots the cdf or pdf for all the features of a network, drawing a line for each subfolder found in input_feature.
    Each subfolder name has to match the pattern "[0-9] - (a-Z)*", for example "1 - Mynetwork". In this case "Mynetwork"
    will be the run name in plot legends.
    :return:
    """
    os.makedirs(output_files+'graphs/PNG', exist_ok=True)
    os.makedirs(output_files+'graphs/PDF', exist_ok=True)

    folders = sorted([f+'/' for f in glob.glob(input_files+"*") if os.path.isdir(f)])
    names = [str.split(os.path.split(f)[0], ' - ')[-1] for f in folders]
    tests_results = [list() for f in folders]
    loaded_data = [load_results_from_files(f,'pandas') for f in folders]


    for fname in Features.features_for_evaluation:
        fig = plt.figure()
        axt = None
        xlabel = [] # Text rows below the graph
        for fold_id, folder in enumerate(folders):

            idx = pd.IndexSlice
            # This is the pandas way to select the same column on each level of a multi-index
            sliced_data = loaded_data[fold_id].loc[idx[:], idx[:, fname]]
            clean_data = sliced_data.dropna(axis=0, how='any')

            tc = clean_data['true'][fname]
            gc = clean_data['gen'][fname]

            # If it's the first network for this feature then plot the true distribution
            if fold_id == 0:
                axt = sb.kdeplot(tc, label="True", ls="--", cumulative=cumulative)
                xlabel.append(fname)
            try:
                sb.kdeplot(gc, label=names[fold_id], cumulative=cumulative)

                # STATISTICAL TESTS
                w_stat, w_pvalue = wilcoxon(tc, gc)
                t_stat, t_pvalue = ttest_ind(tc, gc, nan_policy='omit')
                u_stat, u_pvalue = mannwhitneyu(tc, gc, alternative='two-sided')
                ks_stat, ks_pvalue = ks_2samp(tc, gc)
                tests_results[fold_id].append((fname, w_stat, w_pvalue, t_stat, t_pvalue, u_stat, u_pvalue, ks_stat, ks_pvalue))

                xlabel.append("{} KS stat:{}".format(names[fold_id], ks_stat))
                xlabel.append("{} KS p-value:{}".format(names[fold_id], ks_pvalue))
            except:
                print(
                    "{} matrix is singular and cannot be plotted. \n This is not an error if the feature is not informative with the dataset you are using".format(
                        fname))
                tests_results[fold_id].append((fname, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
        # GRAPH ANNOTATIONS
        axt.set_xlabel("\n".join(xlabel), fontsize=16)
        fig_name = "{}".format(fname)
        fig.canvas.set_window_title(fig_name)
        fig.tight_layout()

        # SAVING THE GRAPH
        fig.savefig(output_files + 'graphs/' + 'PNG/' + fig_name + '.png')
        fig.savefig(output_files + 'graphs/' + 'PDF/' + fig_name + '.pdf')
        plt.close(fig)
        # WRITING STATS TO CSV
    stat_columns = ['feature', 'W-stat', 'W-pvalue', 'T-stat', 'T-pvalue', 'U-stat', 'U-pvalue', 'KS-stat', 'KS-pvalue']

    for n_id, name in enumerate(names):
        with open(output_files + "stat_test_result_{}.csv".format(name), 'w') as csvout:
            writer = csv.writer(csvout, delimiter=',')
            writer.writerow(stat_columns)
            for stat in tests_results[n_id]:
                writer.writerow(stat)


plot_all(cumulative=False)

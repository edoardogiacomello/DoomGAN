import Features
import network_architecture as arch
import network_architecture_WITH_FEATURE as arch_wf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from statsmodels.sandbox.stats.multicomp import multipletests
import os
import csv
import glob
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, ks_2samp


# PATS SPECIFICATION
input_files = './input_files/'
output_files = './output_files/'

to_exclude = [f for f in Features.features_for_evaluation if f.startswith("floors_") or f=='level_bbox_area']

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


def plot_all(cumulative=True, line_styles=['-.', ':']):
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
            try:
                if fold_id == 0:
                    axt = sb.kdeplot(tc, label="True", ls="-", cumulative=cumulative)
                    xlabel.append(fname)

                sb.kdeplot(gc, label=names[fold_id], ls=line_styles[fold_id-1], cumulative=cumulative)

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
        if axt is not None:
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
        with open(output_files + "test_uncorrected_pvalues_{}.csv".format(name), 'w') as csvout:
            writer = csv.writer(csvout, delimiter=',')
            writer.writerow(stat_columns)
            for stat in tests_results[n_id]:
                writer.writerow(stat)

def generate_latex_table(alpha, uncond_columns, stat='KS', correction_method='bonferroni', name_format="test_uncorrected_pvalues_{}.csv", exclude_uninformative_features=True, rejected = 'R', not_rejected = 'N', merge_tables=True):
    """
        Generate text for showing the results associated to the plotted files.
    Loads the stat_test_result for every network and generate a table, highlighting the result if the pvalue is smaller than alpha
    :param alpha:
    :param uncond_columns: a list containing the names of the unconditioned networks (those without input features). The other are considered conditioned.
    :param correction_method:
    :param name_format:
    :param exclude_uninformative_features:
    :param rejected:
    :param not_rejected:
    :return:
    """
    in_folders = sorted([f + '/' for f in glob.glob(input_files + "*") if os.path.isdir(f)])
    names = [str.split(os.path.split(f)[0], ' - ')[-1] for f in in_folders]
    tests_results = list()
    tests_stats = list()



    for name in names:
        test_result_path = output_files + name_format.format(name)
        loaded = pd.read_csv(test_result_path, header=0, index_col=0, comment='#', delimiter=',', usecols=['feature','{}-pvalue'.format(stat),'{}-stat'.format(stat)])
        loaded.rename(columns= {'{}-pvalue'.format(stat): name, '{}-stat'.format(stat): name+'-s'}, inplace=True)
        tests_results.append(loaded[name])
        tests_stats.append(loaded[name+'-s'])


    # Merging dataframes
    data_pv = pd.concat(tests_results, axis=1)
    data_s = pd.concat(tests_stats, axis=1)
    if exclude_uninformative_features:
        data_pv = data_pv.drop(labels=to_exclude, axis=0)
        data_s = data_s.drop(labels=to_exclude, axis=0)

    # Since multipletests relies on the array length for correcting the p-values we concatenate all the columns


    all_pv = np.array([data_pv[col] for col in data_pv.columns]).flatten()
    all_results, all_corr_pv, alpha_sidac, alpha_bonf = multipletests(all_pv, alpha=alpha, method=correction_method)
    nd_all_results = np.split(all_results, len(data_pv.columns))
    nd_all_corr_pv = np.split(all_corr_pv, len(data_pv.columns))
    results = pd.DataFrame().reindex_like(data_pv)
    corr_pv = pd.DataFrame().reindex_like(data_pv)
    for c, col in enumerate(data_pv.columns):
        results[col] = nd_all_results[c]
        corr_pv[col] = nd_all_corr_pv[c]


    # Feature clustering
    def label_features(x, uncond_column, cond_column):
        """
        Compares ONE unconditional network vs ONE conditional network, assigning a label to the feature such that:
        F1: if test is rejected in both networks
        F2: if test is rejected in uncond and not rejected in cond
        F3: if test is not rejected in any case
        F4: if test is not rejected in uncond and rejected in cond
        """

        # F1: Features that are rejected in any case -> "True" in uncond and "True" in wf
        if x[uncond_column] and x[cond_column]:
            return "F1"
        # F2: Features that are rejected in unconditioned and not rejected in some conditioned net -> "True" in uncond and false in wf
        if x[uncond_column] and ~x[cond_column]:
            return "F2"
        # F3: Features that are not rejected in any case
        if ~x[uncond_column] and ~x[cond_column]:
            return "F3"
        # F4: Features that are not rejected in unconditioned and rejected in other cases -> False in uncond, True in wf
        if ~x[uncond_column] and x[cond_column]:
            return "F4"

    def higlight_minimum_distance(x, minimum_values, rej_value=rejected):
        """ Puts an * on the results table if the corresponding feature belong to F1 and the stat is minimal"""
        if (x==rej_value).all():
            min_dist_index = minimum_values[x.name].replace('-s', '')
            return x.where(x.index != min_dist_index, other=x+'*')
        return x



    minimum_stats = data_s.idxmin(axis=1)

    # LABELING FEATURES: Generates a table that has features on the rows and couples of runs on the columns (uncond-vs-cond)
    # For confronting the values we need to have 1 unconditioned vs 1 cond at any time.
    # If there's only one conditioned, just use it for comparing against all the others

    results_uncond = results.loc[:,uncond_columns] # equivalent to results[uncond_columns] but supports assignment
    results_cond = results.drop(uncond_columns, axis=1)
    if len(results_uncond.columns) == 1 and len(results_cond.columns) > 1:
        # Duplicating the cond column for each uncond column then dropping the original one
        for i, uc in enumerate(results_cond.columns):
            results_uncond.insert(loc=i+1, column='{}{}'.format(uncond_columns[0], i),
                                  value=results_uncond[uncond_columns[0]])
        results_uncond.drop(uncond_columns, axis=1, inplace=True)
    if len(results_cond.columns) != len(results_uncond.columns):
        raise ValueError("Cannot compare {} unconditioned runs with {} conditioned runs".format(len(results_cond.columns), len(results_uncond.columns)))
    results_for_labels = pd.concat([results_uncond, results_cond], axis=1)


    feature_labels = pd.DataFrame()
    # Iterating over couples of matching columns
    for uncond_col, cond_col in zip(results_uncond.columns, results_cond.columns):
        feature_labels["{}-vs-{}".format(uncond_col, cond_col)] = results_for_labels[[uncond_col, cond_col]].apply(label_features, args=(uncond_col, cond_col), axis=1)
    results = results.applymap(lambda x: rejected if x else not_rejected)
    results = results.apply(higlight_minimum_distance, axis=1, args=(minimum_stats,))
    if merge_tables:
        results=pd.concat([results, feature_labels.rename(columns={feature_labels.columns[0]:'Group'})], axis=1)
    results_input_feats = results[results.index.isin(arch_wf.features)]
    results_other_feats = results[~results.index.isin(arch_wf.features)]



    f1_features = feature_labels[(feature_labels=='F1').any(axis=1)] # Bad features
    f2_features = feature_labels[(feature_labels=='F2').any(axis=1)] # Improved if some features are in input
    f3_features = feature_labels[(feature_labels=='F3').any(axis=1)] # Good Features
    f4_features = feature_labels[(feature_labels=='F4').any(axis=1)] #

    print("Never working: {}".format(len(f1_features)))
    print("Improved by inputs: {}".format(len(f2_features)))
    print("Always working: {}".format(len(f3_features)))
    print("Deteriorated by inputs: {}".format(len(f4_features)))



    with open(output_files + 'tabular_results.tex', 'w') as out:
        caption_results_input_feats_short = "Test results for input features"
        caption_results_input_feats = "{}-test results for input features, using a significance level of {} and the {} correction method. Results are indicated with R if the null hypotesis can be rejected or with N otherwise. An asterisk indicates the network that performed better (has the minimum KS distance) if the null hypothesis is rejected in every network".format(stat, alpha, correction_method.capitalize())
        caption_results_other_feats_short = "Test results for non input features"
        caption_results_other_feats = "{}-test results for non-input features, using a significance level of {} and the {} correction method. Results are indicated with R if the null hypotesis can be rejected or with N otherwise. An asterisk indicates the network that performed better (has the minimum KS distance) if the null hypothesis is rejected in every network".format(stat, alpha, correction_method.capitalize())
        caption_data_s_short = "{} statistic values".format(stat)
        caption_data_s = "{} statistic values for the tests. The value is correlated with the distance of the cumulative distributions of the true and generated data".format(stat)
        caption_corr_pv_short = "Corrected p-values"
        caption_corr_pv = "Corrected p-values using {} method".format(correction_method.capitalize())
        caption_f1_features_short = "Features belonging to the F1 group"
        caption_f1_features = "Features that belong to group F1. Group F1 contains the features for which the null hypotesis is rejected for both the unconditioned and conditioned network."
        caption_f2_features_short = "Features belonging to the F2 group"
        caption_f2_features = "Features that belong to group F2. Group F2 contains the features for which the null hypotesis is rejected for the unconditioned network and not rejected for the conditioned network."
        caption_f3_features_short = "Features belonging to the F3 group"
        caption_f3_features = "Features that belong to group F3. Group F3 contains the features for which the null hypotesis is not rejected for both the unconditioned and conditioned network."
        caption_f4_features_short = "Features belonging to the F4 group"
        caption_f4_features = "Features that belong to group F4. Group F4 contains the features for which the null hypotesis is not rejected for the unconditioned network and rejected for the conditioned network."

        out.write(results_input_feats.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_results_input_feats_short, caption_results_input_feats), 1).replace('\end{longtable}', ' \\label{tab:results-input-features}\n \end{longtable}', 1))
        out.write(results_other_feats.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n \\label{{tab:results-other-features}}\n'.format(caption_results_other_feats_short, caption_results_other_feats), 1).replace('\end{longtable}', ' \\label{tab:results-input-features}\n \end{longtable}', 1))
        out.write(data_s.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_data_s_short, caption_data_s), 1).replace('\end{longtable}', ' \\label{tab:results-stats}\n \end{longtable}', 1))
        out.write(corr_pv.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_corr_pv_short, caption_corr_pv), 1).replace('\end{longtable}', ' \\label{tab:results-pvalues}\n \end{longtable}', 1))
        out.write(f1_features.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_f1_features_short, caption_f1_features), 1).replace('\end{longtable}', ' \\label{tab:results-f1-features}\n \end{longtable}', 1))
        out.write(f2_features.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_f2_features_short, caption_f2_features), 1).replace('\end{longtable}', ' \\label{tab:results-f2-features}\n \end{longtable}', 1))
        out.write(f3_features.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_f3_features_short, caption_f3_features), 1).replace('\end{longtable}', ' \\label{tab:results-f3-features}\n \end{longtable}', 1))
        out.write(f4_features.to_latex(longtable=True).replace('\n', '\n \\caption[{}]{{ \\small {}}}\\\\\n'.format(caption_f4_features_short, caption_f4_features), 1).replace('\end{longtable}', ' \\label{tab:results-f4-features}\n \end{longtable}', 1))


def generate_latex_figures():
    import itertools
    def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=fillvalue, *args)

    latex=""
    template = """\\begin{{minipage}}[b]{{0.45\\linewidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{results/exp1-2/{file}.pdf}} 
        \\label{{fig:results-input-{file}}}
    \\end{{minipage}}"""

    for feature in arch_wf.features:
        keys = {'feat':feature.replace("_", "\\_"), 'file':feature}
        latex += '\n'+ template.format(**keys)




    with open(output_files+"figures_input.tex", 'w') as out:
        out.write(latex)
    latex_oth = ""
    feat_other = [f for f in Features.features_for_evaluation if f not in arch_wf.features and f not in to_exclude]
    # Iterate over the figures
    for feat_group in grouper(feat_other, 6):
        latex_oth += "\\begin{figure}[ht]\n"
        # Iterate over the rows
        for row in grouper(feat_group, 2):
            latex_row = """\\begin{{minipage}}[b]{{0.45\\linewidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{results/exp1-2/{file1}.pdf}} 
        \\label{{fig:results-noninput-{file1}}}
    \\end{{minipage}}
    \\begin{{minipage}}[b]{{0.45\\linewidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{results/exp1-2/{file2}.pdf}} 
        \\label{{fig:results-noninput-{file2}}}
    \\end{{minipage}} \n \n
                 """
            keys = {'file1':row[0], 'file2':row[1]}
            latex_oth += latex_row.format(**keys)
        latex_oth += "\\caption[Graphical results for experiments 1 and 2]{Experiments 1 and 2: " + \
                         "Cumulative distribution functions for true data, unconditional network and conditional network for each non-input feature.}" + \
                         "\n\\end{figure}"


    with open(output_files + "figures_noninput.tex", 'w') as out:
        out.write(latex_oth)



def plot_tensorboard_from_csv(name_format="run_{net}_{run}-tag-{metric}.csv"):
    raise NotImplementedError()
    cond_loss_train = pd.read_csv("./input_tensorflow_results/run_cond_train-tag-critic_loss.csv" , header=0, comment='#', delimiter=',').rename(columns={'Value': 'Training'})
    cond_loss_valid = pd.read_csv("./input_tensorflow_results/run_cond_validation-tag-critic_loss.csv"    , header=0, comment='#', delimiter=',').rename(columns={'Value': 'Validation'})

    uncond_loss_train =   pd.read_csv("./input_tensorflow_results/run_uncond_train-tag-critic_loss.csv"   , header=0, comment='#', delimiter=',')
    uncond_loss_valid = pd.read_csv("./input_tensorflow_results/run_uncond_validation-tag-critic_loss.csv"  , header=0, comment='#', delimiter=',')

    cond_train = pd.concat([cond_loss_train, cond_loss_valid], axis=1)

    ### WIP - NOT WORKING



if __name__ == '__main__':
    plot_all(cumulative=False)
    generate_latex_table(0.05, uncond_columns=['uncond'], correction_method='bonferroni', stat='KS', merge_tables=True)
    generate_latex_figures()
    #plot_tensorboard_from_csv()
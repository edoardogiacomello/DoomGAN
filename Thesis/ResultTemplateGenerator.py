

text="""\\begin{{figure}}[h!]
	\\begin{{minipage}}{{0.5\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp1/1v1_{0!s}.pdf}}
	\\end{{minipage}}

	\\begin{{minipage}}{{0.5\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp2-12k/1v1_{0!s}.pdf}}
	\\end{{minipage}}
	\\begin{{minipage}}{{0.5\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp2-26k/1v1_{0!s}.pdf}}
	\\end{{minipage}}

	\\begin{{minipage}}{{0.5\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp3-12k/1v1000_{0!s}.pdf}}
	\\end{{minipage}}
	\\begin{{minipage}}{{0.5\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp3-26k/1v1000_{0!s}.pdf}}
	\\end{{minipage}}
	\\caption[ Results: Input feature {1!s}]{{ Results of the experiments 1, 2 and 3 for the feature {1!s}. \\\\* Experiment 1 (first row): True distribution (red, dashed) vs Generated distribution (blue, solid) in the case of a network with no input features. \\\\* Experiment 2 (second row): True distribution (red, dashed) vs Generated distribution (blue, solid) in the case of a conditional network trained for 12000 (left) and 26000 (right) iterations. \\\\* Experiment 3 (third row): Input feature values for the 25th (red), 50th (green), 75th (blue) percentile vs. the corresponding generated distribution}}
	\\label{{fig:results_{0!s}}}
\\end{{figure}}"""

text_oth ="""\\begin{{figure}}[h!]
    \\centering
	\\begin{{minipage}}{{0.4\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp1/1v1_{0!s}.pdf}}
	\\end{{minipage}}

	\\begin{{minipage}}{{0.4\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp2-12k/1v1_{0!s}.pdf}}
	\\end{{minipage}}
	\\begin{{minipage}}{{0.4\\linewidth}}
		\\includegraphics[width=\\linewidth]{{results/exp2-26k/1v1_{0!s}.pdf}}
	\\end{{minipage}}

	\\caption[ Results: Feature {1!s}]{{ Results of the experiments 1 (first row) and 2 (second row) for the feature {1!s}. Refer to the results section for further descriptions. }}
	\\label{{fig:appendix_{0!s}}}
\\end{{figure}}"""

features = {
'level_equivalent_diameter',
'level_major_axis_length',
'level_minor_axis_length',
'level_solidity',
'nodes',
'distmap-skew',
'distmap-kurt'
}

other_features = {
    'level_area',
    'level_bbox_area',
    'level_convex_area',
    'level_eccentricity',
    'level_euler_number',
    'level_extent',
    'level_filled_area',
    'level_orientation',
    'level_perimeter',
    'level_hu_moment_0',
    'level_hu_moment_1',
    'level_hu_moment_2',
    'level_hu_moment_3',
    'level_hu_moment_4',
    'level_hu_moment_5',
    'level_hu_moment_6',
    'level_centroid_x',
    'level_centroid_y',
    'number_of_artifacts',
    'number_of_powerups',
    'number_of_weapons',
    'number_of_ammunitions',
    'number_of_keys',
    'number_of_monsters',
    'number_of_obstacles',
    'number_of_decorations',
    'walkable_area',
    'walkable_percentage',
    'start_location_x_px',
    'start_location_y_px',
    'artifacts_per_walkable_area',
    'powerups_per_walkable_area',
    'weapons_per_walkable_area',
    'ammunitions_per_walkable_area',
    'keys_per_walkable_area',
    'monsters_per_walkable_area',
    'obstacles_per_walkable_area',
    'decorations_per_walkable_area',
    'avg-path-length',
    'diameter-mean',
    'art-points',
    'assortativity-mean',
    'betw-cen-min',
    'betw-cen-max',
    'betw-cen-mean',
    'betw-cen-var',
    'betw-cen-skew',
    'betw-cen-kurt',
    'betw-cen-Q1',
    'betw-cen-Q2',
    'betw-cen-Q3',
    'closn-cen-min',
    'closn-cen-max',
    'closn-cen-mean',
    'closn-cen-var',
    'closn-cen-skew',
    'closn-cen-kurt',
    'closn-cen-Q1',
    'closn-cen-Q2',
    'closn-cen-Q3',
    'distmap-max',
    'distmap-mean',
    'distmap-var',
    'distmap-Q1',
    'distmap-Q2',
    'distmap-Q3'
}
for n, f in enumerate(other_features):
    with open('raw_latex_results.tex', 'a') as tex:
        if n > 0 and n % 2 == 0:
            tex.write("\n \\newpage \n")
        tex.write("\n \n % Results for {} \n".format(f))
        tex.write(text_oth.format(f, f.replace("_", "\\textunderscore ")))

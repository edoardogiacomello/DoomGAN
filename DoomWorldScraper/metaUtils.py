import dicttoxml
import json
import csv
import os
from collections import defaultdict
import zipfile
from WADParser import WADParser, WADRasterizer

header_to_dict_index = defaultdict(lambda: "other", {
            'Author': 'author',
            'About This File': 'description',
            'Credits': 'credits',
            'Base': 'base',
            'Editors Used': 'editor_used',
            'Bugs': 'bugs',
            'Build Time': 'build_time',
            # These headers does not appear as is on the website, but they are kept here for keeping track of all the
            # possible columns
            'rating_value': 'rating_value',
            'rating_count': 'rating_count',
            'page_visits': 'page_visits',
            'downloads': 'downloads',
            'creation_date': 'creation_date',
            'file_url': 'file_url',
            'game': 'game',
            'category': 'category',
            'title':'title',
            'name:':'name',
            'path':'path'
        })


def json_to_xml(path):
    """Converts a json file to xml"""
    with open(path, 'r') as jfile:
        j = json.load(jfile)
    xml = dicttoxml.dicttoxml(j, attr_type=False, custom_root='category', item_func=lambda x: 'WAD')
    dest_path = path.replace('.json','.xml') if path.endswith('.json') else path+'.xml'
    with open(dest_path, 'w') as dst_file:
        dst_file.write(xml.decode('utf-8'))


def json_to_csv(path):
    """Converts a json file containing a list of entries in a csv file"""
    with open(path, 'r') as jfile:
        j = json.load(jfile)
        dest_path = path.replace('.json','.csv') if path.endswith('.json') else path+'.csv'
    with open(dest_path, 'w') as csvfile:
        keys = header_to_dict_index.values()
        dict_writer = csv.DictWriter(csvfile, keys)
        dict_writer.writeheader()
        dict_writer.writerows(j)

def merge_json(file_list, output_path):
    """Take a list of filenames as input (e.g. ./Doom<CATNAME>.json) and merge all the files in a single
    file located at output_path."""
    final = []
    for file in file_list:
        with open(file, 'r') as jfile:
            j = json.load(jfile)
            final += j
    with open(output_path, 'w') as joutput:
        json.dump(final, joutput)
    print("Merged {} records in one file".format(len(final)))


def extract_database(root_path, json_filename, output_path):
    """This function takes the folder of scraped data as input (the folder structure should be
        <root_path>/<catname>/levelname.zip) and generates a representation that is similar to that used by
        https://github.com/TheVGLC/TheVGLC (i.e. <gamename>/<[Original|Processed|Processed - Vectorized]>)
         at a given output_path. In particular it extract each level zip file and parse them in both tile and vector 
         representations. The json file passed as input is used as a reference for finding the files corresponding
        to the levels, and it is updated while the WADs get parsed.
        
        root_path: the folder containing the categories subfolders and the json_file
        json_file: metadata database for the levels that are contained in root_path. 
                   Should be located at <root_path>/<json_file>
        output_path: the output folder for extraction. Should be an empty folder as this function may overwrite sensitive data
        
        EXAMPLE: extract_database("./dataset/doom/","DoomIncomplete.json", "./dataset/doom/parsed/")
        """
    # TODO: Continue this
    assert os.path.exists(root_path), "Specified directory does not exist."
    assert os.path.isfile(root_path+json_filename), "Specified json file does not exist."
    supported_extensions = ('.wad', '.bws')

    path_output_json_bad = output_path+'check_manually.json'
    path_output_json_good = output_path + json_filename

    path_output_original = output_path + 'Original/'
    path_output_processed = output_path + 'Processed/'
    path_output_vectorized = output_path + 'Processed - Vectorized/'

    # Creates the output paths
    if not os.path.exists(path_output_original):
        os.makedirs(path_output_original)
    if not os.path.exists(path_output_processed):
        os.makedirs(path_output_processed)
    if not os.path.exists(path_output_vectorized):
        os.makedirs(path_output_vectorized)

    parsed_levels = list() # Contain all the successfully parsed levels


    # Load the json file
    with open(root_path+json_filename, 'r') as jf_in:
        level_records = json.load(jf_in)

    check_manually = []
    for level in level_records:
        path = level['path']
        wads = []
        try:
            with zipfile.ZipFile(path) as zipped:
                wads += filter(lambda x: x.filename.lower().endswith(supported_extensions), zipped.infolist())
                if len(wads) == 0:
                    print("{} does not contain any supported file.".format(path))
                    # TODO: Inspect nested zip files.
                    check_manually.append(level)
                for wadinfo in wads:
                    # Extract the wad into the Original folder
                    zipped.extract(wadinfo, path=path_output_original)
                    level['path'] = path_output_original+wadinfo.filename
                    # Parse and Raserize the levels contained in this wad
                    try:
                        rasterized_levels = WADRasterizer.rasterize(level['path'], output=path_output_processed)
                        vectorized_levels = WADParser.parse(level['path'], output=path_output_vectorized)
                    except (UnicodeDecodeError, ValueError, TypeError) as e:
                        print("Cannot decode " + level['path'])
                        check_manually.append(level)
                        continue

                    for tile_rep, svg_rep in zip(rasterized_levels, vectorized_levels):
                        parsed_level = level
                        parsed_level['tile_path'] = tile_rep
                        parsed_level['svg_path'] = svg_rep
                        parsed_levels.append(parsed_level)

                print("{}: extracted {} levels from {} wad files".format(zipped.filename, len(rasterized_levels), len(wads)))
        except (zipfile.BadZipFile, NotImplementedError) as e:
            print("Found a malformed file: {}".format(path))
            check_manually.append(level)

    with open(path_output_json_good, 'w') as json_out:
        json.dump(parsed_levels, json_out)
    print("Saved {} levels to".format(path_output_json_good))

    with open(path_output_json_bad, 'w') as json_out:
        json.dump(check_manually, json_out)
    print("You have {} levels to check manually. They are found at {}".format(len(check_manually), path_output_json_bad))


# extract_database("./database/doom/","DoomIncomplete.json", "./database/doom/parsed/")
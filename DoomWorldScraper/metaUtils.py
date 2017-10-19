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
            'path':'path',
            'svg_path':'svg_path',
            'tile_path':'tile_path'
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


def extract_database(json_database, output_path):
    """This function takes the folder of scraped data as input (the folder structure should be
        <root_path>/<catname>/levelname.zip) and generates a representation that is similar to that used by
        https://github.com/TheVGLC/TheVGLC (i.e. <gamename>/<[Original|Processed|Processed - Vectorized]>)
         at a given output_path. In particular it extract each level zip file and parse them in both tile and vector 
         representations. The json file passed as input is used as a reference for finding the files corresponding
        to the levels, and it is updated while the WADs get parsed.

        json_database: metadata database for the levels that are contained in root_path.
                   Must be located at <root_path>/<json_database> along with the category-named folders
        output_path: the output folder for extraction. Should be an empty folder as this function may overwrite sensitive data
        
        EXAMPLE: extract_database("./dataset/doom/","DoomIncomplete.json", "./dataset/doom/parsed/")
        """
    supported_extensions = ('.wad', '.bws')

    def flatten_zip(file):
        """Extracts wad files from nested zips"""
        wads = []
        zips = []
        with zipfile.ZipFile(file) as zipped:
            try:
                 for z in zips:
                    wads += flatten_zip(z)
            except Exception as e:
                print("Couldn't extract the zip file.")
                print(str(e))
            return wads

    assert os.path.isfile(json_database), "Specified json file does not exist."

    path_output_json_bad = output_path+'check_manually.json'
    path_output_json_good = output_path + json_database.split('/')[-1]

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
    with open(json_database, 'r') as jf_in:
        level_records = json.load(jf_in)

    check_manually = []
    for level in level_records:
        path = level['path']
        wads = []
        zips = []
        if not (path.endswith('.zip') and os.path.isfile(path)):
            continue
        try:
            with zipfile.ZipFile(path) as zipped:
                wads += filter(lambda x: x.filename.lower().endswith(supported_extensions), zipped.infolist())
                zips += filter(lambda x: x.filename.lower().endswith('.zip'), zipped.infolist()) #Nested Zips
                zipname = path.split('/')[-1].split('.')[0]
                if len(zips)>0:
                    print("{} has {} nested zip files inside".format(path, len(zips)))
                for wadinfo in wads:
                    # Extract the wad into the Original folder
                    zipped.extract(wadinfo, path=path_output_original)
                    extracted_path = path_output_original+wadinfo.filename
                    # We prepend the zip file name to the parsed level name to avoid overwriting levels with the same wad name
                    target_path = path_output_original+zipname+'_'+wadinfo.filename
                    # Sometimes wads come inside folders, so the path must exists
                    if not os.path.exists('/'.join(target_path.split('/')[0:-1])+'/'):
                        os.makedirs('/'.join(target_path.split('/')[0:-1])+'/')
                    os.rename(extracted_path, target_path)
                    level['path'] = target_path

                    # Parse and Raserize the levels contained in this wad
                    try:
                        rasterized_levels = WADRasterizer.rasterize(level['path'], output=path_output_processed)
                        vectorized_levels = WADParser.parse(level['path'], output=path_output_vectorized)
                        parsed_counter = 0
                        for tile_rep, svg_rep in zip(rasterized_levels, vectorized_levels):
                            parsed_level = level
                            parsed_level['tile_path'] = tile_rep
                            parsed_level['svg_path'] = svg_rep
                            parsed_levels.append(parsed_level.copy())
                            parsed_counter += 1
                        print("{} levels parsed".format(parsed_counter))
                    except Exception as e:
                        print("Cannot decode " + level['path'])
                        check_manually.append(level)
                        continue
        except (zipfile.BadZipFile, NotImplementedError) as e:
            print("Found a malformed file: {}".format(path))
            print(str(e))
            check_manually.append(level)

    with open(path_output_json_good, 'w') as json_out:
        json.dump(parsed_levels, json_out)
    print("Saved {} levels to {}".format(len(parsed_levels), path_output_json_good))

    with open(path_output_json_bad, 'w') as json_out:
        json.dump(check_manually, json_out)
    print("You have {} level files to check manually. They are found at {}".format(len(check_manually), path_output_json_bad))


def remove_duplicates(path):
    with open(path,'r') as inf:
        j = json.load(inf)
    valid_paths = list()
    result = list()
    dupes = 0
    for level in j:
        if not level['tile_path'] in valid_paths:
            valid_paths.append(level['tile_path'])
            result.append(level)
        else:
            dupes += 1
    with open(path+'fix', 'w') as ouf:
        json.dump(result, ouf)
    print("Removed {} duplicates of {} records".format(dupes, len(result)))

def check_processed_folders(json_path, root):
    with open(json_path,'r') as inf:
        j = json.load(inf)
    tile_paths = [r['tile_path'] for r in j]
    tile_folder = root+'Processed/'
    for file in os.listdir(tile_folder):
        if tile_folder+file not in tile_paths:
            print("{} has no entries in json file".format(file))

def check_consistency(json_path, root):
    with open(json_path, 'r') as inf:
        j = json.load(inf)
    result = []
    json_paths = [r['path'] for r in j]
    for folder in os.listdir(root):
        if os.path.isdir(root+folder+'/'):
            for file in os.listdir(root+folder+'/'):
                fullpath = root + folder + '/' + file
                if fullpath.endswith(".zip"):
                    if fullpath in json_paths:
                        result.append(fullpath)
    print(str(len(j)))
    print(str(len(result)))



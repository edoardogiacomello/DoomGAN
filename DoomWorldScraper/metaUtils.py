import dicttoxml
import json
import csv
from collections import defaultdict
import zipfile

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

def extract_database(root_path):
    """This function takes the folder of scraped data as input (the folder structure should be
        <root_path>/<gamename>/<catname>/levelname.zip) and generates a representation that is similar to that used by
        https://github.com/TheVGLC/TheVGLC at a given output_path.
        The json file passed as input is updated accordingly."""
    # TODO: Continue this
    pass

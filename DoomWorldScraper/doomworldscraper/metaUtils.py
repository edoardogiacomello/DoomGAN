import dicttoxml
import json
import csv
from collections import defaultdict

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
    with open(path, 'r') as jfile:
        j = json.load(jfile)
    xml = dicttoxml.dicttoxml(j, attr_type=False, custom_root='category', item_func=lambda x: 'WAD')
    dest_path = path.replace('.json','.xml') if path.endswith('.json') else path+'.xml'
    with open(dest_path, 'w') as dst_file:
        dst_file.write(xml.decode('utf-8'))


def json_to_csv(path):
    with open(path, 'r') as jfile:
        j = json.load(jfile)
        dest_path = path.replace('.json','.csv') if path.endswith('.json') else path+'.csv'
    with open(dest_path, 'w') as csvfile:
        keys = header_to_dict_index.values()
        dict_writer = csv.DictWriter(csvfile, keys)
        dict_writer.writeheader()
        dict_writer.writerows(j)

def merge_json(file_list, output_path):
    final = []
    for file in file_list:
        with open(file, 'r') as jfile:
            j = json.load(jfile)
            final += j
    with open(output_path, 'w') as joutput:
        json.dump(final, joutput)
    print("Merged {} records in one file".format(len(final)))        


json_file_list = ['./database/Doom0-9.json',
                  './database/Dooma-c.json',
                  './database/Doomd-f.json',
                  './database/Doomg-i.json',
                  './database/Doomj-l.json',
                  './database/Doomm-o.json',
                  './database/Doomp-r.json',
                  './database/Dooms-u.json',
                  './database/Doomv-z.json']

merged_path = './database/Doom.json'

#merge_json(json_file_list, merged_path)
#json_to_csv(merged_path)

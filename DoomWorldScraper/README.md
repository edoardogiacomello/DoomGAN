# Dataset Pipeline
This folder contains the scraper and the utilities I wrote to build the dataset.

Here follow some instructions in case you need to replicate the building process or if you want to further expand the dataset.

Notice that each of this step could take a lot of time (even several hours) depending on your resources.
#### Main Steps:
1) ##### Scrape some levels. (Check Python Doc  for more details)
```
from DoomWorldScraper.DoomWorldWadScraper import DoomWorldWadScraper
scraper = DoomWorldWadScraper()
scraper.collect_wads(game_category_url, game_name="Doom", exclude_category_list=[68, 85, 87], root_path="./scraped/doom/")
```
This will create a folder "./scraped/doom/" containing a "Doom.json" file and several folders with the same names as the doomworld categories.
These folders contain the zip files that have been downloaded from the website, while Doom.json contain data collected from the download page.

2) ##### Extract and parse the dataset
```
    from metaUtils import extract_database
    extract_database('./scraped/doom/', './dataset/')
```
This checks for every zip file addressed by the Doom.json database looking for .WAD files and extract them 
into the *./dataset/Original/* folder. For each WAD, it tries to read each level, doing the following:
- Parse the [features](https://docs.google.com/spreadsheets/d/1Lv6fVyk_7QaZRpwhSvgRVB9EXe3KcNt1UV5cZx3RS24/edit?usp=sharing)
- Save all its scalar/string features to  *./dataset/Processed/\<zipname\>\_<LEVELNAME\>_\<level_slot>.json*
- Save all feature features to *./dataset/Processed/\<zipname\>\_<LEVELNAME\>_\<level_slot>\_\<map_name>.json*
- The content that is saved to each individual .json is also stored as a row of *./dataset/Processed/Doom.json*


3) ##### Cluster and filter the data
    This is up to you and your needs. You should filter a list of records from the .json database file (Doom.json) in the example and pass to the next step

4) ##### Pack the data to a  .TFRecords file
    Recently Tensorflow introduced a new standard for storing and reading data from files. It also seems to speedup data ingestion compared to other method such reading from separate files. 
    ``` 
    import dataset_utils.convert_to_TFRecords(record_list, output_path, max_size, min_size=(16, 16), normalize_in=(0,1))
    ```
    This function normalizes your data based on the range of the features and packs the levels in a <output_path>.TFRecord file.
     It also stores a <output_path>.meta needed for keeping data about the features that are relative to the selected subset of data, such max/min/avg features and the encoding 
    obtained after normalization.
5) ##### Reading the dataset
    A tensorflow dataset object is available by simply calling 
    ```
    dataset_utils.load_TFRecords_database(path)
    ```
    Notice that this needs a .meta file in the same folder of the .TFRecords file, since otherwise it wouldn't be possible to decode image data without knowing anything about the size.
    
    Please refer to the Tensorflow documentation to proceed further.
    
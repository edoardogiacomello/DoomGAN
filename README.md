# DoomPCGML
Main Repository for my MSc thesis about Procedural content generation via machine learning applied to videogames levels (Doom I/II)


## Useful Links
* [Log Progressi (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) my personal log where I note the current state of the project and what is left to do, with a brief explaination of each step.
* [Bibliography (it)](https://drive.google.com/open?id=1yZST4OQBGEaHU4lrI0jtZQuYQ6ts4mLapJRkP17iMw0) Sources and documents that have been useful to the project
* [Overview (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) Brief description of the project 
and possible developments
* [Features](https://docs.google.com/spreadsheets/d/1Lv6fVyk_7QaZRpwhSvgRVB9EXe3KcNt1UV5cZx3RS24/edit?usp=sharing) Features collected for each level


## Repo Folders Explaination:
The project is composed of several modules that can be used separately.

Each module has a dedicated readme, if you need further information you can inspect the code documentation and comments since I try to keep the code as clear as possible. 
- [DoomWorldScraper](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper) contains a scraper for www.doomworld.com (which is an ID Software Level Archive mirror) from which the Community Generated levels have been collected.
- [DoomWorldScraper/WADParser](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper/WADParser) modified version of the parser present at the VGLC repo for better integration and automation with the scraper.
- [GAN-Tests/DCGAN-slerp-interpolation](https://github.com/edoardogiacomello/DoomPCGML/tree/master/GAN-Tests/DCGAN-slerp-interpolation) Some experiments and implementations I made to DCGAN architecture about visualizing and operating on the latent space for understanding how to apply them to Doom Levels. See the corresponding readme.

### Requirements (WIP)
A complete list of python packages will come soon, these are the most important requirements
* Python 3.*
* Tensorflow with cuda support
* numpy
* skimage

## Dataset
The dataset is currently composed of about 9600 already parsed levels and it's temporarly stored in the /dataset/ folder. 
Metadata about the levels are stored in corresponding json files but it's also available in google sheets for a more convenient consultation:
- Levels from DoomI/II taken from @TheVGLC [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- Levels from "Final Doom" and "Master levels for DoomII" taken from a retail steam copy of the game. [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- About 9600 levels for DoomI and DoomII taken from www.doomworld.com along with their metadata [preview here](https://drive.google.com/open?id=1LeKRCot5eu69_y71w8EYIBYQdwkDtdWlx68nmLqTYxg)
- There are still more than 2000 levels to download, but since they are for co-op/deathmatch they could present structural differences with standard levels so I decided to let them out of the dataset for now.

The latest dataset is stored at: [Google Sheets](https://drive.google.com/open?id=1MB61Gt-xfp_obJy4nlf5NpnRKb0rt-C1G14Os3uVbQI)


### Dataset Pipeline
The dataset comes from the Id archive, an archive of levels from ID Software. Data has been scraped from www.doomworld.com in date 16/10/2017.

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
    

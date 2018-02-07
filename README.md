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
You can find all the requirements in the requirements.txt file. The most important are:
* Python 3.*
* tensorflow_gpu 1.3 with cuda 8.0 support (more recent versions of TF may work but have not been tested)
* tensorboard (software)
* numpy
* scipy
* scikit_image
* scikit_learn
* matplotlib
* networkx
* pandas
* seaborn



#### Optional requirements
* beautifulsoup4 for the scraper
* dicttoxml for the scraper
* requests for the scraper

* bsp (Shell command) for generating WAD files
## Dataset
The full dataset comes from the Id archive, an archive of levels from ID Software. 
Data has been scraped from www.doomworld.com in date 16/10/2017, obtaining about 9600 levels.
Metadata about the levels are stored in corresponding json files but it's also available in google sheets for a more convenient consultation:
- Levels from DoomI/II taken from @TheVGLC [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- Levels from "Final Doom" and "Master levels for DoomII" taken from a retail steam copy of the game. [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- About 9600 levels for DoomI and DoomII taken from www.doomworld.com along with their metadata [preview here](https://drive.google.com/open?id=1LeKRCot5eu69_y71w8EYIBYQdwkDtdWlx68nmLqTYxg)
- There are still more than 2000 levels to download, but since they are for co-op/deathmatch they could present structural differences with standard levels so I decided to let them out of the dataset for now.

The latest dataset is stored at: [Google Sheets](https://drive.google.com/open?id=1MB61Gt-xfp_obJy4nlf5NpnRKb0rt-C1G14Os3uVbQI)


The full dataset is structured as follow:
- a root _<dataset_root>/_
- a .json database _<dataset_root>/dataset.json_ in which all the level features are stored
- a _<dataset_root>/Original/_ folder, containing the .WAD files for each level
- a _<dataset_root>/Processed/_ folder, containing:
    - <zip_name>_<wad_name>_<slot_name>.json file containing the features for a level (one row dataset.json)
    - <zip_name>_<wad_name>_<slot_name>_<feature_map>.png image(s) containing the feature map for a level
    
These files are indexed from the dataset.json starting from _<dataset_root>_.  
E.g. a path could be _"Processed/myzip_MyLevel_E1M1_floormap.png"_  
The feature maps dimensions are (width/32, height/32), since 32 is the diameter of the smallest thing that exists on Doom.  
Each pixel value is an uint8, dicrectly encoding a value (ie. the "thing type index" for thingsmap; 1,2,3,4.. for
the "floormap" enumeration or the floor_height value for the heightmap.

Dataset is also stored in a .TFRecord file (and this is the format DoomGAN uses to read the dataset);
this is useful if you want to previously filter a dataset perhaps selecting only 128x128 levels and padding smaller ones.  
This way you pack all the dataset in a single .TFRecord file and its relative .meta file, containing statistics 
for each feature, such as min/max/avg value along the samples that have been selected in order to further normalize the data.

Into the "dataset" folder you can find two datasets:
- 128x128-many-floors.zip: Train and validation set for levels up to 128 pixels with any number of floors
- 128x128-one-floor.zip: Train and validation set for levels up to 128 pixels with just one floor
Each file comes with a .meta file that is the one to reference in the script parameters.

## Running the code

Before running the model you should make sure that the datasets are extracted into the "dataset" folder. Zip files containing
datasets for 128x128 levels are already included in the repo, you only have to unzip them:
``` 
cd dataset
unzip 128x128-one-floor.zip
 ```
 this will produce two .TFRecord and one .meta files. The .meta file is the one you have to specify as parameter if you need it.
 
 Run scripts for most common operations with default parameters are in the project root. 
 Make sure of setting your current folder (cd) because it will be used by the script for adding it to the pythonpath.
 
 For training a network you just need to issue:    
 ``` ./train.sh ```
 
 This command will start the training and save a new network model into the ```artifacts/checkpoint``` folder every 50 iterations.
 You can stop the training any time and resume it by running the script again.
 
 If you want to monitor the progress and check the current network output you can issue:
 ``` tensorboard --logdir=<project_root>/artifacts/tensorboard_results/ ``` (remembering to subsitute <project_root> with the appropriate folder) and open the returned link in your browser.
 
 If you need to redo the training  from scratch you can manually delete the content of the artifacts folder, or launch the script ```./reset.sh```
 
 If you want to generate new samples just run  ```./generate.sh``` and the output will be produced in the artifact folder.
 
 If you need to change the network structure (layers or input maps/features) just edit the file `DoomLevelsGAN/network_architecture.py` remembering of resetting the Artifacts folder before training the new model.
 
 

 
 
 
 


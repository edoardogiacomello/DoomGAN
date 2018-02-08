# DoomPCGML
Main Repository for my MSc thesis about Procedural content generation via machine learning applied to videogames levels (Doom I/II)


## Useful Links
* [Log Progressi (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) my personal log where I note the current state of the project and what is left to do, with a brief explaination of each step.
* [Bibliography (it)](https://drive.google.com/open?id=1yZST4OQBGEaHU4lrI0jtZQuYQ6ts4mLapJRkP17iMw0) Sources and documents that have been useful to the project
* [Overview (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) Brief description of the project 
and possible developments
* [Features](https://docs.google.com/spreadsheets/d/1Lv6fVyk_7QaZRpwhSvgRVB9EXe3KcNt1UV5cZx3RS24/edit?usp=sharing) Features collected for each level


### Requirements
#### Using docker
Your host should be able to run:
* Docker
* Nvidia-Docker (https://github.com/NVIDIA/nvidia-docker)
* Cuda 8.0
* Tensorboard (optional)

#### Using your own environment
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


## Running the code
### Using Docker

1) Install Cuda 8.0
2) Install Docker
3) Install nvidia-docker from https://github.com/NVIDIA/nvidia-docker
4) Move (cd) to an empty folder where you want to store the results
5) Launch the Docker (WIP)

This commnand will open a shell inside the DoomPCGML docker and will create an "artifacts" folder in your current directory that will contain:
* Network checkpoints (your network model, saved every 100 iterations -> ~16 minutes or less)
* Tensorboard cache (training progress data).  
* Generated Samples


### Manual method
Running the code directly in your python environment (without Docker or virtualenv) is simple, you only have to make sure of having a 
 working installation of cuda 8, tensorflow and python 3.5 or above.

Clone the repository with

``` git clone --depth=1 https://github.com/edoardogiacomello/DoomPCGML.git ```

and install the remaining dependencies with
 
 ``` pip install --user -r requirements.txt ```

follow the "usage" section.

 
 ## Usage
 ***README*** Run scripts for most common operations with default parameters are in the project root. 
 Make sure of calling them from the project root itself (you have to "cd" into DoomPCGML folder if you have just cloned the repository) becuse some scripts makes use of the current directory
 for importing all the modules.
 
 ### Extracting the dataset
 Before running the model you should make sure that the datasets are extracted into the "dataset" folder. Zip files containing
datasets for 128x128 levels are already included in the repository, you only have to unzip them with:

``` ./extract.sh ```

 this will produce two .TFRecord and one .meta files. The .meta file is the one you have to specify as parameter if you need it.
 
 **_you have to repeat the process every time you launch the container if you are using docker_**
 
 ###Train the network
 For training a network (or resuming a training session) you just need to issue:    
 ``` ./train.sh ```

 This command will start the training and save a new network model into the ```artifacts/checkpoint``` folder every 100 iterations.

 #### Stopping the training  ``` ctrl + c ```
 
 #### Detaching/Attaching from/to the container shell (Docker)
 If you need to detach from the container shell without stopping the training process you have to press `ctrl+p` followed by `ctrl+q`.
 In order to reattach to the shell you can issue `nvidia-docker attach <container_name>`. You can find the current name with the `docker ps` command
. 

 #### Restarting from scratch
 If you need to redo the training from scratch (for example if you modified the network architecture) you have to manually delete the following folders:
 `/artifacts/tensorboard_results` and `/artifacts/checkpoints`. This will every model and its relative training data. 
  
 #### Sampling the network
 If you want to generate new samples from your current network just run  ```./generate.sh``` and the output will be produced in the artifact folder.
 
 #### Changing inputs and network structure
 If you need to change the network structure (layers or input maps/features) just launch ``` ./edit_network.sh``` (nano required) or edit the file `DoomLevelsGAN/network_architecture.py` remembering to reset the Artifacts folder before training the new model.
 
 #### Visualizing training progress
 ##### If you have tensorboard in the host machine
 This method has the advantage that you don't have to stop the training process to launch tensorboard:
 Just issue ``` tensorboard --logdir=<cache_folder>/artifacts/tensorboard_results``` where <cache_folder> is the folder you cloned the repo into or the container volume mountpoint if you are using docker.
 ##### Using the version included in the container (Docker)
 You have run your container with an additional parameter: `-p 6006:6006`, then run the command above on the container shell. 
 
 ## Advanced usage
 The commands explained above are for basic "black box" usage, but the script have lots of option you can configure (such as the sampling operation). 
 
 You can edit the "train" script adding several parameters for testing different input size, learning hyperparameters and so on.
 
 ###Parameter List
 _WIP_
 
 ###Outputing to WAD files
 You can generate working WAD files directly from the nework samples. Documentation is still in progress but you can give it a try by checking the DoomGAN.py script and running the sample function with 'WAD' save parameter.
 
 ## Performances
 The developement machine had the following specs:
 - Intel i7 940
 - Nvidia GTX670 4GB
 - 18GB Ram
 
 and it took about 10 seconds per iteration with the default network architecture with the following usages:
 * VRAM:  3826MiB / 4035MiB
 * RAM: 1.77 GiB / 17.63 GiB
 
 So you are goind to need a machine with at least 4GB of VRAM to run the code. I also noticed some overhead when using Docker.

 
 

 
 ## Dataset
_**The full dataset is currently unavailable online, it will be uploaded as soon as possible. (See below.)**_

The full dataset comes from the Id archive, an archive of levels from ID Software. 
Data has been scraped from www.doomworld.com in date 16/10/2017, obtaining about 9600 levels.
Metadata about the levels are stored in corresponding json files but it's also available in google sheets for a more convenient consultation:
- Levels from DoomI/II taken from @TheVGLC [Provided as a separated dataset]
- Levels from "Final Doom" and "Master levels for DoomII" taken from a retail steam copy of the game. [Provided as a separated dataset]
- About 9600 levels for DoomI and DoomII taken from www.doomworld.com along with their metadata [preview here](https://drive.google.com/open?id=1LeKRCot5eu69_y71w8EYIBYQdwkDtdWlx68nmLqTYxg)
- There are still more than 2000 levels to download, but since they are for co-op/deathmatch they could present structural differences with standard levels so I decided to let them out of the dataset for now.

A dataset preview and a list of features stored at: [Google Sheets](https://drive.google.com/open?id=1MB61Gt-xfp_obJy4nlf5NpnRKb0rt-C1G14Os3uVbQI)


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

**_Dataset is currently provided as a .TFRecord file ready to be ingested by DoomGAN_**    

Into the "dataset" folder you can find two datasets:
- 128x128-many-floors.zip: Train and validation set for levels up to 128 pixels with any number of floors
- 128x128-one-floor.zip: Train and validation set for levels up to 128 pixels with just one floor
Each file comes with a .meta file that is the one to reference in the script parameters.

 
## Repo Folders Explaination:
The project is composed of several modules that can be used separately.

Each module has a dedicated readme, if you need further information you can inspect the code documentation and comments since I try to keep the code as clear as possible. 
- [DoomWorldScraper](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper) contains a scraper for www.doomworld.com (which is an ID Software Level Archive mirror) from which the Community Generated levels have been collected.
- [DoomWorldScraper/WADParser](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper/WADParser) modified version of the parser present at the VGLC repo for better integration and automation with the scraper.
- [GAN-Tests/DCGAN-slerp-interpolation](https://github.com/edoardogiacomello/DoomPCGML/tree/master/GAN-Tests/DCGAN-slerp-interpolation) Some experiments and implementations I made to DCGAN architecture about visualizing and operating on the latent space for understanding how to apply them to Doom Levels. See the corresponding readme.

 


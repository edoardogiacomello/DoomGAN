# DoomPCGML
Main Repository for my MSc thesis about Procedural content generation via machine learning applied to videogames levels (Doom I/II)

## Useful Links
* [Log Progressi (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) my personal log where I note the current state of the project and what is left to do, with a brief explaination of each step.
* [Bibliography (it)](https://drive.google.com/open?id=1yZST4OQBGEaHU4lrI0jtZQuYQ6ts4mLapJRkP17iMw0) Sources and documents that have been useful to the project
* [Overview (it)](https://drive.google.com/open?id=1mEVQyaKdYL0TJLRg279CoGagp1Bz2aU7lSQNQvCc1Wc) Brief description of the project 
and possible developments

## Dataset
The dataset is currently composed of about 9600 already parsed levels and it's temporarly stored in the /dataset/ folder. 
Metadata about the levels are stored in corresponding json files but it's also available in google sheets for a more convenient consultation:
- Levels from DoomI/II taken from @TheVGLC [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- Levels from "Final Doom" and "Master levels for DoomII" taken from a retail steam copy of the game. [preview](https://drive.google.com/open?id=1SUbK48BSfG_qSyxokkWr-R0XlT_D40OlCoRljevKbmM)
- About 9600 levels for DoomI and DoomII taken from www.doomworld.com along with their metadata [preview here](https://drive.google.com/open?id=1LeKRCot5eu69_y71w8EYIBYQdwkDtdWlx68nmLqTYxg)
- There are still more than 2000 levels to download, but since they are for co-op/deathmatch they could present structural differences with standard levels so I decided to let them out of the dataset for now.

The latest dataset is stored at: [Google Sheets](https://drive.google.com/open?id=1MB61Gt-xfp_obJy4nlf5NpnRKb0rt-C1G14Os3uVbQI)

## Repo Folders Explaination:
- [DoomWorldScraper](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper) contains a scraper for www.doomworld.com (which is an ID Software Level Archive mirror) from which the Community Generated levels have been collected.
- [DoomWorldScraper/WADParser](https://github.com/edoardogiacomello/DoomPCGML/tree/master/DoomWorldScraper/WADParser) modified version of the parser present at the VGLC repo for better integration and automation with the scraper.
- [GAN-Tests/DCGAN-slerp-interpolation](https://github.com/edoardogiacomello/DoomPCGML/tree/master/GAN-Tests/DCGAN-slerp-interpolation) Some experiments and implementations I made to DCGAN architecture about visualizing and operating on the latent space for understanding how to apply them to Doom Levels. See the corresponding readme.

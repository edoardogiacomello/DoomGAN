import requests
import os
import json
from collections import namedtuple
from collections import defaultdict
from bs4 import BeautifulSoup
from bs4 import Tag

class DoomWorldWadScraper():
    """Collects WAD data present at www.doomworld.com
    This class is compatible to the website layout as of 10/10/2017"""

    def __init__(self):
        self.tuple_category = namedtuple("category",["name","url", "file_pages"])
        self.categories = dict()
        self.file_info = list()
        # This dict is to convert what is written to the webpage to the indices that are used in the json representation
        self.header_to_dict_index = defaultdict(lambda: "other", {
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
            'title': 'title',
            'name:': 'name',
            'path': 'path'
        })

    def _open_page(self, url):
        """Set up a BeautifulSoup for the given page"""
        self.session = requests.Session()
        page = self.session.get(url).text
        return BeautifulSoup(page, "lxml")

    def fetch_categories(self, url, exclude_list=[]):
        """Fetches every Download subcategory entry on doomworld.com starting from a game link like
        https://www.doomworld.com/files/category/64-doom/ and proceeds for just one level down the tree. 
        The ids in exclude_cat_id are skipped """
        soup = self._open_page(url)
        links = soup.find_all('a')
        for l in links:
            if l.has_attr('href') and "/category/" in l['href'] and l.find("strong") is not None:
                url = l['href']
                id = l['href'].split('/')[-2].split('-')[0]
                if not isinstance(l.contents[1], Tag) or int(id) in exclude_list:
                    # The considered element is not at the next nevel. e.g. ./deathmatch/a-z while on /levels/
                    # because we found the name of the class instead of a tag that contains it.
                    continue
                name = l.contents[1].contents[0]
                self.categories[id] = self.tuple_category(name=name, url=url, file_pages=[])
        return self.categories

    def _fetch_file_page_links(self, category_url):
        """Fetches, for each category page, the list of the download pages that holds the data for each wad file by crawling
        through each list page, recursively.
        i.e. from an url of the type https://www.doomworld.com/files/category/<CATEGORYNAME>/?page=N
        navigates through all the N pages returning all the file urls like
        https://www.doomworld.com/files/file/<file-id>-<file-name>/"""
        found_files = []
        print ("Opening category page: " + category_url)
        soup = self._open_page(category_url)
        # Find the div containing the pagination design pattern and the file links
        mainDiv = soup.findAll("div", {"id": "ipsLayout_mainArea"})[0]
        file_page_links = [entry.find_all("a")[0] for entry in mainDiv.find_all("div", {"class":"ipsDataItem_main"})]
        found_files = found_files + [fpl['href'] for fpl in file_page_links]
        print("Found {} more files".format(str(len(found_files))))

        # Check if there are more pages to crawl
        next_page_buttons = mainDiv.find_all("li", {"class" : "ipsPagination_next"})
        if len(next_page_buttons)>0 and "ipsPagination_inactive" not in next_page_buttons[0]['class']:
            new_category_url = next_page_buttons[0].a['href']
            print("The list continues to the page " + new_category_url)
            next_pages_files = self._fetch_file_page_links(new_category_url)
            found_files += next_pages_files
        return found_files

    def _fetch_download_page_data(self, download_page_url, game_name=None, category_name=None):
        """Returns a dict containing all the useful data present in a download page"""

        # The data is stored in a defaultdict since not all the entries may be present on the page
        download_data = defaultdict(str)
        soup = self._open_page(download_page_url)
        # We focus only on the frame that contains the data
        main_div = soup.find_all("div", {"id" : "ipsLayout_mainArea"})[0].find_all("div",{'itemscope':'', 'itemtype':"http://schema.org/CreativeWork"})[0]
        rating_meta = main_div.div.find_all("meta")

        # These tags contain the main article with the headers for Title, Author, Description, Credits, Build Time..
        level_info_tag_heads = [s.text.strip() for s in main_div.article.div.contents[1:-2:4]]  # These are the section heas (Author, About this file..)
        level_info_tag_content = [s.text.strip() for s in main_div.article.div.contents[3:-2:4]] # The corresponding content

        # Setting the title
        download_data['title'] = main_div.div.div.contents[1].div.text



        for header, content in zip(level_info_tag_heads, level_info_tag_content):
            download_data[self.header_to_dict_index[header]] = content

        # Adding the metadata relative to the file itself (always present)
        file_information_panel = main_div.aside.div
        file_meta = file_information_panel.find_all("meta")

        download_data['rating_value'] = rating_meta[0]["content"]
        download_data['rating_count'] = rating_meta[1]["content"]
        download_data['page_visits'] = file_meta[0]["content"].split(":")[-1]
        download_data['downloads'] = file_meta[1]["content"].split(":")[-1]
        download_data['creation_date'] = file_meta[2]["content"]
        download_data['file_url'] = file_information_panel.a['href']
        download_data['game'] = game_name
        download_data['category'] = category_name

        return download_data

    def _download_and_save(self, url, local_path):
        """Downloads the file at 'url' to 'local_path' and returns the full path of the downloaded file or None on failure"""
        r = self.session.get(url)
        if r.status_code is not 200:
            print ("Failed to download the file: " + url)
            return None
        filename = r.url.split('/')[-1]
        with open(local_path+filename, "wb") as file:
            file.write(r.content)
        return local_path+filename

    def collect_wads(self, game_category_url, game_name, exclude_category_list=[], root_path = "./"):
        """Execute the pipeline for collecting all the wad files present in a category page."""

        # Check if the json file is present and try to resume downloading if possible
        json_path = root_path + game_name + '.json'
        if os.path.isfile(json_path):
            print("Trying to resume download...")
            with open(json_path, 'r') as jsonfile:
                self.file_info = json.load(jsonfile)
                print("Loaded {} records.".format(len(self.file_info)))

        self.fetch_categories(game_category_url, exclude_list=exclude_category_list)
        # For each category found, collect the list of files and populate the relative tuple entry
        for cat_id in self.categories.keys():
            # Creates a new local folder if not present
            current_path = root_path + self.categories[cat_id].name + '/'

            if not os.path.exists(current_path):
                os.makedirs(current_path)

            current = 1
            current_records = []
            visited_links = ['/'.join(x.split('/')[0:-1])+"/" for x in [d['file_url'] for d in self.file_info]]
            links = self._fetch_file_page_links(self.categories[cat_id].url)

            for link in links:
                try:
                    # Check if the list already contains the link we are trying to download
                    if link in visited_links:
                        print("[{}/{}] . {} already present in the dataset, skipping...".format(current, len(links), link))
                        current +=1
                        continue
                    print("[{}/{}] - Collecting {}".format(current, len(links), link))
                    current += 1
                    newfile = self._fetch_download_page_data(link,
                                                             game_name=game_name,
                                                             category_name=self.categories[cat_id].name)
                    # download the level file and store it on the local disk
                    file_path = self._download_and_save(newfile['file_url'], current_path)
                    # Remove the root from the path
                    newfile['path'] = file_path.replace(root_path, './')
                    newfile['name'] = newfile['path'].split("/")[-1]
                    current_records.append(newfile)
                    self.file_info.append(newfile)

                    # Save the scraped data in case of abnormal interruption
                    if current % 5 == 0:
                        with open(json_path, 'w') as jsonfile:
                            json.dump(self.file_info, jsonfile)
                        print("Records saved to {}...".format(json_path))
                except Exception:
                    print("Couldn't download {}".format(link))


        with open(json_path, 'w') as jsonfile:
            json.dump(self.file_info, jsonfile)
        print("{} records saved to {}. Done.".format(len(self.file_info), json_path))



if __name__ == '__main__':
    # Create a scraper for "Home>Downloads>idgames>levels>doom"
    scraper = DoomWorldWadScraper()
    # The link structure of each category page is: https://www.doomworld.com/files/category/##-???/
    # where ## is a numerical id (eg. 64 for "doom") and ??? is the category name, such as "a-c"

    # We are going to skip the "Deathmatch" (68), "Megawads" (85) and "Ports" (87) categories since they will be dealt with separately.
    scraper.collect_wads(game_category_url='https://www.doomworld.com/files/category/64-doom/', game_name="Doom", exclude_category_list=[68, 85, 87], root_path="./dataset/doom/")
    # scraper.collect_wads(game_category_url='https://www.doomworld.com/files/category/100-doom2/', game_name="DoomII", exclude_category_list=[104, 129, 131, 101], root_path="./dataset/doomII/")
    import metaUtils
    metaUtils.extract_database('./dataset/doom/Doom.json', './WADs/Doom/')
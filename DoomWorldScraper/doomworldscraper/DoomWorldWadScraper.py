from bs4 import BeautifulSoup
from bs4 import Tag
import requests
from collections import namedtuple

class DoomWorldWadScraper():
    """Collects WAD data present at www.doomworld.com
    This class is compatible to the website layout as of 10/10/2017"""

    def __init__(self, baseurl):
        self.page = requests.get(baseurl).text
        self.soup = BeautifulSoup(self.page, "lxml")
        self.tuple_category = namedtuple("category",["name","url"])
        self.categories = dict()

    def fetch_categories(self, exclude_list=[]):
        """Fetches every Download subcategory entry on doomworld.com starting from a game link like
        https://www.doomworld.com/files/category/64-doom/ and proceeds for just one level down the tree. 
        The ids in exclude_cat_id are skipped """
        links = self.soup.find_all('a')
        for l in links:
            if l.has_attr('href') and "/category/" in l['href'] and l.find("strong") is not None:
                url = l['href']
                id = l['href'].split('/')[-2].split('-')[0]
                if not isinstance(l.contents[1], Tag) or int(id) in exclude_list:
                    # The considered element is not at the next nevel. e.g. ./deathmatch/a-z while on /levels/
                    # because we found the name of the class instead of a tag that contains it.
                    continue
                name = l.contents[1].contents[0]
                self.categories[id] = self.tuple_category(name=name, url=url)
        return self.categories

    def fetch_list_pages(self):
        """Fetches, for each category page, the list of the download pages that hold the data for each wad file by crawling
        through each list page, recursively."""


# Create a scraper for "Home>Downloads>idgames>levels>doom"
scraper = DoomWorldWadScraper(baseurl='https://www.doomworld.com/files/category/64-doom/')
# The link structure of each category page is: https://www.doomworld.com/files/category/##-???/
# where ## is a numerical id (eg. 64 for "doom") and ??? is the category name, such as "a-c"

# We are going to skip the "Deathmatch" (68), "Megawads" (85) and "Ports" (87) categories since they will be dealt with separately.
cats = scraper.fetch_categories(exclude_list=[68, 85, 87])
for id in cats.keys():
    print(cats[id].name)
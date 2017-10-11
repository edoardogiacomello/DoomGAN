import scrapy

class DoomWorldSpider(scrapy.Spider):
    name = "downloads"

    def start_requests(self):
        urls = [
            'https://www.doomworld.com/files/category/63-levels/'
        ]
        for url in urls:
            yield scrapy.Request(url=urls, callback=self.parse)

    def parse(self, response):
        for link in response.xpath('//a[contains(@href,"category")]'):
            pass
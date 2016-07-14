# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html

from scrapy.item import Item, Field

class CraigslistSampleItem(Item):
    title = Field()
    link = Field()

class PhItem(Item):
    title = Field()
    link = Field()
    id = Field()
    duration = Field()
    file_urls = Field()
    file_paths = Field()
    tags = Field()
    viewkey = Field()
    tags = Field()
    thumbnails = Field()
    thumbsFrequency = Field()
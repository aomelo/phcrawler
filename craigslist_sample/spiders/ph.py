from scrapy.spiders import CrawlSpider, Rule
from scrapy import Request
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
from craigslist_sample.items import PhItem
import re
from demjson import decode
from math import ceil
import craigslist_sample.settings
from craigslist_sample.settings import VIDEO_RESOLUTION

class MySpider(CrawlSpider):
    name = "ph"
    prefix = "http://www.pornhub.com"
    allowed_domains = ["www.pornhub.com",
                       "cdnt4b.video.pornhub.phncdn.com",
                       "cdn1.video.pornhub.phncdn.com",
                       "cdn2b.video.pornhub.phncdn.com",
                       "[0-9|a-z|A-Z|.]*.rncdn3.com",
                       ".*.video.pornhub.phncdn.com",]
    start_urls = ["http://www.pornhub.com/video?c=41",
                  #"http://www.pornhub.com/channels/povd/videos?o=vi",
                  #"http://www.pornhub.com/channels/povd/videos?o=ra"
                  ]

    #rules = (
    #    Rule(LinkExtractor(allow=(), restrict_xpaths=('//li[@class="page_next"]',)), callback="parse", follow=True),
    #)

    def generate_file_urls(self,item):
        num_urls = int(ceil(float(item["duration"])/float(item["thumbsFrequency"])/25.0))
        file_urls = []
        pattern = item["thumbnails"]
        print pattern
        for i in range(num_urls):
            url = re.sub("S\{[0-9]*\}", ("S"+str(i)), pattern)
            print url
            file_urls.append(url)
        return file_urls



    def parse_video(self, response):
        hxs = Selector(response)

        item = PhItem()
        item["link"] = response.url
        item["viewkey"] = re.search("viewkey=(.*)", response.url).group(1)
        item["id"] = hxs.xpath('//div/@data-video-id').extract()[0]
        item["title"] = hxs.xpath('//title').extract()[0]
        item["duration"] = hxs.xpath('//div/@data-duration').extract()[0]
        jscode = hxs.xpath('//div[@id="player"]/script[@type="text/javascript"]').extract()[0]
        if not jscode == []:
            #download_url = re.search("var player_quality_"+settings.VIDEO_RESOLUTION+" = '(.*)';", jscode[0]).group(1).split(";")[0]
            #download_url = re.search("var player_quality_"+VIDEO_RESOLUTION+"p = '(.*)';", jscode).group(1).split(";")[0]
            #if download_url and VIDEO_RESOLUTION in download_url:
            #    item["file_urls"] = [download_url.replace("'","")]
            jscode = hxs.xpath('//div[@class="video-wrapper"]/div/script[@type="text/javascript"]').extract()[0]
            flash_vars = re.search("var flashvars_[0-9]* = (\{.*\});",jscode).group(1)
            jsonvars = decode(flash_vars)
            if "actionTags" in jsonvars:
                tags = jsonvars["actionTags"]
                if tags:
                    item["tags"] = tags
                    item["thumbnails"] = jsonvars["thumbs"]["urlPattern"]
                    item["thumbsFrequency"] = jsonvars["thumbs"]["samplingFrequency"]
                    height = int(jsonvars["thumbs"]["thumbHeight"])
                    width = int(jsonvars["thumbs"]["thumbWidth"])
                    if height==90 and width>=160 and item["thumbnails"]:
                        item["file_urls"] = self.generate_file_urls(item)
                        yield item

    def parse(self, response):
        hxs = Selector(response)
        videos = hxs.xpath('//li[@class="videoblock"]/div/div/a')
        if not videos:
            videos = hxs.xpath('//div[@class="phimage"]/a')
        for video in videos:
            url = response.urljoin(video.xpath("@href").extract()[0])
            yield Request(url, callback=self.parse_video, method="GET")
        next_page = hxs.xpath('//li[@class="page_next"]/a/@href')
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield Request(url, callback=self.parse, method="GET")
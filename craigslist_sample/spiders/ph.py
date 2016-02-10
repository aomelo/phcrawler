from scrapy.spiders import CrawlSpider, Rule
from scrapy import Request
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
from craigslist_sample.items import PhItem
import re
from demjson import decode
import craigslist_sample.settings

class MySpider(CrawlSpider):
    name = "ph"
    prefix = "http://www.pornhub.com"
    allowed_domains = ["www.pornhub.com",
                       #"cdnt4b.video.pornhub.phncdn.com",
                       "cdn1.video.pornhub.phncdn.com",
                       "cdn2b.video.pornhub.phncdn.com",
                       "[a-z|0-9].rncdn3.com",
                       "[a-z|0-9]*.video.pornhub.phncdn.com",]
    start_urls = ["http://www.pornhub.com/channels/brazzers/videos?o=da",
                  #"http://www.pornhub.com/channels/povd/videos?o=ra"
                  ]

    #rules = (
    #    Rule(LinkExtractor(allow=(), restrict_xpaths=('//li[@class="page_next"]',)), callback="parse", follow=True),
    #)

    def parse_video(self, response):
        hxs = Selector(response)

        item = PhItem()
        item["link"] = response.url
        item["id"] = hxs.xpath('//div[@class="video-wrapper"]/div/@data-video-id').extract()
        item["title"] = hxs.xpath('//title').extract()
        item["duration"] = hxs.xpath('//div/@data-duration').extract()[0]
        jscode = hxs.xpath('//div[@id="player"]/script[@type="text/javascript"]').extract()
        if not jscode == []:
            #download_url = re.search("var player_quality_"+settings.VIDEO_RESOLUTION+" = '(.*)';", jscode[0]).group(1).split(";")[0]
            download_url = re.search("var player_quality_240p = '(.*)';", jscode[0]).group(1).split(";")[0]
            if download_url:
                item["file_urls"] = [download_url.replace("'","")]
                jscode = hxs.xpath('//div[@class="video-wrapper"]/div/script[@type="text/javascript"]').extract()
                flash_vars = re.search("var flashvars_[0-9]* = (\{.*\});",jscode[0]).group(1)
                jsonvars = decode(flash_vars)
                if "actionTags" in jsonvars:
                    tags = jsonvars["actionTags"]
                    if tags:
                        item["tags"] = tags
                        yield item

    def parse(self, response):
        hxs = Selector(response)
        videos = hxs.xpath('//li[@class="videoblock"]/div/div/a')
        for video in videos:
            url = response.urljoin(video.xpath("@href").extract()[0])
            yield Request(url, callback=self.parse_video, method="GET")
        next_page = hxs.xpath('//li[@class="page_next"]/a/@href')
        if next_page:
            url = response.urljoin(next_page[0].extract())
            yield Request(url, callback=self.parse, method="GET")
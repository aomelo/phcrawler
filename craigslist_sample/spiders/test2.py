from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from craigslist_sample.items import CraigslistSampleItem


class MySpider(CrawlSpider):
    name = "craigs"
    allowed_domains = ["sfbay.craigslist.org"]
    start_urls = ["http://sfbay.craigslist.org/search/npo"]

    rules = (
        Rule(LinkExtractor(allow=(), restrict_xpaths=('//a[@class="button next"]',)), callback="parse_items", follow= True),
    )

    def parse_items(self, response):
        print "\n\n\n\n" + response.url + "\n\n\n\n"
        hxs = HtmlXPathSelector(response)
        titles = hxs.xpath('//span[@class="pl"]')
        print hxs.xpath('//a[@class="button next"]').extract()
        items = []
        for titles in titles:
            item = CraigslistSampleItem()
            item["title"] = titles.xpath("a/text()").extract()
            item["link"] = titles.xpath("a/@href").extract()
            items.append(item)
        return(items)

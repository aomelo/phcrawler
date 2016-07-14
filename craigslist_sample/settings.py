# Scrapy settings for craigslist_sample project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#

BOT_NAME = 'craigslist_sample'

DOWNLOAD_HANDLERS = {'s3': None}

SPIDER_MODULES = ['craigslist_sample.spiders']
NEWSPIDER_MODULE = 'craigslist_sample.spiders'
ITEM_PIPELINES = {'craigslist_sample.pipelines.Mp4Pipeline': 1#,
                  #'craigslist_sample.pipelines.XmlExportPipeline': 2
                  }
DOWNLOADER_MIDDLEWARES = {
    'craigslist_sample.middlewares.PhMiddleware': 543,
}
FILES_STORE = './data/'
FILES_EXPIRES = 7
SAMPLE_INTERVAL_SEC = 5
VIDEO_RESOLUTION = '240' # 240p,480p,720p,1080p
COOKIES_ENABLED = False
DOWNLOAD_DELAY = 1
DOWNLOAD_MAXSIZE = 257741824
DOWNLOAD_WARNSIZE = 107741824
DOWNLOAD_TIMEOUT = 600

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'craigslist_sample (+http://www.yourdomain.com)'

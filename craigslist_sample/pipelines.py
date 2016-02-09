# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/topics/item-pipeline.html

from scrapy.pipelines.files import FilesPipeline
from scrapy import Request
from scrapy.exceptions import DropItem
from scrapy.xlib.pydispatch import dispatcher
from scrapy import signals
from scrapy.contrib.exporter import XmlItemExporter
import cv2
import os
import settings

class CraigslistSamplePipeline(object):
    def process_item(self, item, spider):
        return item


class PhItemPipeline(object):
    def process_item(self, item, spider):
        return item

class Mp4Pipeline(FilesPipeline):

    #def get_media_requests(self, item, info):
    #    for file_url in item['file_urls']:
    #        yield Request(file_url)

    def get_tags_dict(self, tags):
        tagsdict = {}
        for entry in tags.split(","):
            sec = int(entry.split(":")[1])
            tag = entry.split(":")[0]
            tagsdict[sec] = tag
        return tagsdict


    def item_completed(self, results, item, info):
        file_paths = [x['path'] for ok, x in results if ok]
        if not file_paths:
            raise DropItem("Item contains no files")

        item['file_paths'] = file_paths
        for path in file_paths:

            tagsdict = self.get_tags_dict(item['tags']);


            print path
            path = settings.FILES_STORE +path
            vid = cv2.VideoCapture(path)
            id = item["id"][0]

            if not os.path.isdir("None"):
                os.mkdir("None")
            for tag in tagsdict.values():
                if not os.path.isdir(tag):
                    os.mkdir(tag)

            success = True
            sec = 0
            success,image = vid.read()
            while success:
                tag = "None"
                index = [k for k in tagsdict if k <= sec]
                if index:
                    tag = tagsdict[max(index)]
                cv2.imwrite(tag+"/vid"+id+"frame"+str(sec)+".jpg", image)
                sec = sec + settings.SAMPLE_INTERVAL_SEC
                vid.set(0,sec*1000) # 0 = CAP_PROP_POS_MSEC
                success,image = vid.read()
            os.remove(path)



        return item

    #def process_item(self, item, spider):
    #    info = self.spiderinfo
    #    requests = arg_to_iter(self.get_media_requests(item, info))
    #    dlist = [self._process_request(r, info, item) for r in requests]
    #    dfd = DeferredList(dlist, consumeErrors=1)
    #    return dfd.addCallback(self.item_completed, item, info)


class XmlExportPipeline(object):

   def __init__(self):
       dispatcher.connect(self.spider_opened, signals.spider_opened)
       dispatcher.connect(self.spider_closed, signals.spider_closed)
       self.files = {}

   def spider_opened(self, spider):
       file = open('%s_items.xml' % spider.name, 'w+b')
       self.files[spider] = file
       self.exporter = XmlItemExporter(file)
       self.exporter.start_exporting()

   def spider_closed(self, spider):
       self.exporter.finish_exporting()
       file = self.files.pop(spider)
       file.close()

   def process_item(self, item, spider):
       self.exporter.export_item(item)
       return item
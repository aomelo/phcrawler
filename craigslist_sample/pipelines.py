# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/topics/item-pipeline.html

from scrapy.pipelines.files import FilesPipeline
from scrapy import Request
from scrapy.exceptions import DropItem
from scrapy.xlib.pydispatch import dispatcher
from scrapy import signals
from scrapy.contrib.exporter import JsonItemExporter
import numpy as np
import cv2
import os
import settings
from scipy import ndimage
from scipy.misc import imsave
from settings import FILES_STORE, VIDEO_RESOLUTION, SAMPLE_INTERVAL_SEC

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
    image_height = VIDEO_RESOLUTION
    image_width = 426

    def get_tags_dict(self, tags):
        tagsdict = {}
        for entry in tags.split(","):
            sec = int(entry.split(":")[1])
            tag = entry.split(":")[0]
            tagsdict[sec] = tag
        return tagsdict


    def crop(self, image):
        diff_height = image.shape[0] - self.image_height
        diff_width = image.shape[1] - self.image_width
        offset_x = int(diff_height/2)
        offset_y = int(diff_width/2)
        return image[offset_x:offset_x+image_height,offset_y:offset_y+image_width]

    def split_thumbnails(self, image, rows=5, cols=5):
        thumbs = []
        h = image.shape[0]
        w = image.shape[1]
        t_h = h/rows
        t_w = w/cols
        for i in range(rows):
            for j in range(cols):
                thumbs.append(image[i*t_h:(i+1)*t_h , j*t_w:(j+1)*t_w])
        return thumbs

    def item_completed(self, results, item, info):
        file_paths = [x['path'] for ok, x in results if ok]
        if not file_paths:
            raise DropItem("Item contains no files")

        tagsdict = self.get_tags_dict(item['tags']);
        none_tag = settings.FILES_STORE+"None"
        if not os.path.isdir(none_tag):
            os.mkdir(none_tag)
        for tag in tagsdict.values():
            tag = settings.FILES_STORE + tag
            if not os.path.isdir(tag):
                os.mkdir(tag)

        thumbs = []
        for path in file_paths:
            thumb25_path = os.path.join(FILES_STORE,path)
            thumb25 = ndimage.imread(thumb25_path)
            thumbs = thumbs + self.split_thumbnails(thumb25,5,5)
            thumb25 = None
            os.remove(thumb25_path)

        output_paths = []
        interval = int(item["thumbsFrequency"])
        duration = int(item["duration"])
        sec = 0
        for thumb in thumbs:
            if sec<duration:
                tag = "None"
                index = [k for k in tagsdict if k <= sec]
                if index:
                    tag = tagsdict[max(index)]
                image_name = item["viewkey"]+"-frame"+str(sec)+".jpg"
                image_path = os.path.join(FILES_STORE,tag,image_name)
                imsave(image_path, thumb)
                output_paths.append(image_path)
                sec += interval
        item["file_paths"] = output_paths







    def item_completed_vid(self, results, item, info):
        file_paths = [x['path'] for ok, x in results if ok]
        if not file_paths:
            raise DropItem("Item contains no files")

        output_paths = []
        for path in file_paths:
            tagsdict = self.get_tags_dict(item['tags']);
            vid_path = os.path.join(FILES_STORE,path)
            vid = cv2.VideoCapture(vid_path)
            id = item["id"]

            none_tag = settings.FILES_STORE+"None"
            if not os.path.isdir(none_tag):
                os.mkdir(none_tag)
            for tag in tagsdict.values():
                tag = settings.FILES_STORE + tag
                if not os.path.isdir(tag):
                    os.mkdir(tag)

            print item['duration']
            duration = int(item['duration'])

            success = True
            sec = 0
            success,image = vid.read()
            #if image.shape[0] != self.image_height or image.shape[1] < self.image_width:
            #   return None
            print image.shape


            while success and sec <= duration:
                #if image.shape[1] > self.image_width:
                #   image = crop(image)

                tag = "None"
                index = [k for k in tagsdict if k <= sec]
                if index:
                    tag = tagsdict[max(index)]
                image_name = item["viewkey"]+"-frame"+str(sec)+".jpg"
                image_path = os.path.join(FILES_STORE,tag,image_name)
                cv2.imwrite(image_path, image)
                output_paths.append(image_path)
                sec = sec + SAMPLE_INTERVAL_SEC
                vid.set(0,sec*1000) # 0 = CAP_PROP_POS_MSEC
                success,image = vid.read()
            vid.release()
            os.remove(vid_path)

        item['file_paths'] = output_paths
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
        self.exporter = JsonItemExporter(file)
        self.exporter.start_exporting()

    def spider_closed(self, spider):
        self.exporter.finish_exporting()
        file = self.files.pop(spider)
        file.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item
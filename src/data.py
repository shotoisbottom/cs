from random import shuffle
import os
import skimage
import numpy as np
import json

class IOdata():
    def __init__(self, directory, batchs_size, label=None, dirname = None, dataRandow=True):
        if(dirname == None):
            self.queue = [d for d in os.listdir(directory)
                        if d.endswith([".png", ".jpg"])]
        else:
            self.queue = [d for d in os.listdir(os.path.join(directory,dirname[0]))
                        if d.endswith([".png", ".jpg"])]
        if label:
            self.label = self.read_json(label)
        self.dataRandow = dataRandow
        self.index = 0
        self.batchs_size = batchs_size
        self.name = dirname
        self.directory = directory
        self.reset()

    def read_json(label_path):
        with open(label_path, "r") as json_file:
            data = json.load(json_file)
        image_list = [image["file_name"] for image in data["images"]]
        label_list = [label["category_id"] for label in data["annotations"]]
        result = {}
        for k,v in zip(image_list, label_list):
            result[k] = int(v)
        return result

    def reset(self):
        if(self.dataRandow):
            shuffle(self.queue)
        self.index = 0

    def read_file(self,path):
        directories = self.queue[self.index:self.index + self.batchs_size]
        images = []
        for d in directories:
            images.append(skimage.io.imread(os.path.join(path,d)).astype(np.float32)/255)
        return images

    def getsize(self):
        return len(self.queue)

    def imageAndLabel(self):
        file = []
        label = []

        if(self.name == None):
            file.append(self.read_file(self.directory))
            label.append(self.label[self.directory])
        else:
            for d in self.name:
                file.append(self.read_file(os.path.join(self.directory,d)))
                label.append(self.label[d])

        self.index = self.index+self.batchs_size

        if(self.index == self.getsize()):
            self.reset()

        return file, label

    def __call__(self, *args,  **kwds):
        file = []

        if(self.name == None):
            file.append(self.read_file(self.directory))
        else:
            for d in self.name:
                file.append(self.read_file(os.path.join(self.directory,d)))

        self.index = self.index+self.batchs_size

        if(self.index == self.getsize()):
            self.reset()

        return file

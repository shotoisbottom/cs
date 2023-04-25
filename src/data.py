from random import shuffle
import os
import skimage
import numpy as np

class IOdata():
    def __init__(self, directory, batchs_size, dirname = None, dataRandow=True):
        if(dirname == None):
            self.queue = [d for d in os.listdir(directory)
                        if d.endswith([".png", ".jpg"])]
        else:
            self.queue = [d for d in os.listdir(os.path.join(directory,dirname[0]))
                        if d.endswith([".png", ".jpg"])]
        self.dataRandow = dataRandow
        self.index = 0
        self.batchs_size = batchs_size
        self.name = dirname
        self.directory = directory
        self.reset()

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

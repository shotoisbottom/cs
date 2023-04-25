import tensorflow as tf 
import numpy as np
import configparser
import json
import time
import os

from models import UnetProMax as Net
from data import IOdata
from tools.io import data_resize

filename = 'config.conf'
modelpath = 'weights/a.h5'

config = configparser.ConfigParser()
config.read(filename)

shape = json.loads(config.get('Data', 'shape'))
data_dir = config.get('Data', 'data_dir')
save_dir = config.get('Data', 'save_dir')

Data = IOdata(data_dir, batchs_size=1, dirname='test')

model = Net(name="val")
model.load_weights(filename)
model.build(input_shape=[1, shape[0], shape[1], shape[2]])
model.summary(print_fn=print)

def end(x):
    return tf.maximum(0.0,tf.minimum(x,1.0))

def imsave(img,name,pth):
    img = tf.io.encode_png(img)
    tf.io.write_file(os.path.join(pth,name),img)

for _ in range(0,Data.getsize()): 
    x,y = Data()
    x,y = data_resize(x, y, shape=[shape[0], shape[1]])
    pred = model(x)
    image = tf.concat([x,y,pred],axis=2)
    save = imsave(save,str(_) +".png",save_dir)


import tensorflow as tf 
import numpy as np
import configparser
import json
import time

from models import UnetProMax as Net
from data import IOdata
from tools.io import data_resize

filename = 'config.conf'

config = configparser.ConfigParser()
config.read(filename)

network_name = config.get('Network', 'name')
num_epochs = config.getint('Network', 'num_epochs')
init_rate = config.getfloat('Network', 'init_rate')
loss = config.get('Network', 'Loss')

batch_size = config.getint('Data', 'batchSize')
shape = json.loads(config.get('Data', 'shape'))
data_dir = config.get('Data', 'data_dir')
save_dir = config.get('Data', 'save_dir')

trainData = IOdata(data_dir, batchs_size=batch_size, label=True, dirname='inputs')
valData = IOdata(data_dir, batchs_size=1, dirname='val')


def evaluate_yolov5_accuracy(yolo_model, test_data, ground_truth_labels):
    pred_label = yolo_model(test_data)
    return tf.keras.metrics.binary_accuracy(pred_label, ground_truth_labels)


model = Net(name=network_name)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=init_rate),
    loss=tf.keras.losses.mae,
    metrics=tf.image.psnr
    )
model.build(input_shape=[batch_size, shape[0], shape[1], shape[2]])
optimizer = tf.keras.optimizers.Adam(learning_rate=init_rate)
model.summary(print_fn=print)

detect = tf.keras.models.load_model(config.get('TeacherNet', 'path'))

def loss_object(x=None, y=None, label=None):
    loss = tf.keras.losses.mae(x,y)
    if label == None:
        return loss
    else:
        return loss*evaluate_yolov5_accuracy(yolo_model=detect, test_data=x, ground_truth_labels=label)

def train_step(x=None, y=None, label=None):
    with tf.GradientTape() as tape:
        pred = model(x,training=True)
        loss = loss_object(pred,y,label)
    gradients = tape.gradient(loss, model.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred, tf.reduce_mean(loss)

def evaluate_step(x=None, y=None):
    pred = model(x,training=False)
    loss = loss_object(pred,y)
    return pred, tf.reduce_mean(loss)

for _ in range(num_epochs):
    for batch in range(0,trainData.getsize()//batch_size): 
        x,y,l = trainData.imageAndLabel()
        x,y = data_resize(x, y, shape=[shape[0], shape[1]])
        l = tf.concat(l,axis=-1)
        pred, loss = train_step(x, y, l)
        print("train_step loss: %d"%(loss))

    x,y = valData()
    x,y = data_resize(x, y, shape=[shape[0], shape[1]])
    pred, loss = evaluate_step(x, y)
    print("epoch %d/%d, loss: %d"%(_, num_epochs, loss))

end_time = time.strftime('%m%d_%H%M%S', time.localtime())
model.save_weights("weights"+str(end_time)+".h5")
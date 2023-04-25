import tensorflow as tf 
import numpy as np
import configparser
import json
import cv2

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

trainData = IOdata(data_dir, batchs_size=batch_size, dirname='inputs')
valData = IOdata(data_dir, batchs_size=1, dirname='val')

def preprocess_image(image, input_size):
    image = cv2.resize(image, (input_size, input_size))
    image = np.float32(image)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def post_process_output(predictions, anchors, classes, input_size, image_size, conf_thresh=0.25, iou_thresh=0.45):
    # Post-processing code goes here
    pass

def evaluate_yolov5_accuracy(yolo_model, test_data, ground_truth_labels):
    # Initialize counters
    total_predictions = 0
    correct_predictions = 0
    
    # Iterate through test images and ground truth labels
    for image, ground_truth in zip(test_data, ground_truth_labels):
        # Preprocess the image
        preprocessed_image = preprocess_image(image, shape)
        
        # Run inference
        predictions = yolo_model(preprocessed_image)
        
        # Post-process the output to obtain the predicted bounding boxes and class labels
        boxes, scores, classes = post_process_output(predictions, anchors, classes, shape, image.shape[:2])
        
        # Calculate recognition accuracy
        # ...
        # Update total_predictions and correct_predictions counters based on your comparison of predictions and ground_truth
        
    # Calculate and return recognition accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


model = Net(name=network_name)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=init_rate),
    loss=tf.keras.losses.mae,
    metrics=tf.image.psnr
    )
model.build(input_shape=[batch_size, shape[0], shape[1], shape[2]])

detect = tf.keras.models.load_model()

for _ in range(num_epochs):
    x,y = trainData()
    x,y = data_resize(x, y, shape=[shape[0], shape[1]])
    model.fit(x, y, batch_size=batch_size, epochs=1)

    x,y = valData()
    x,y = data_resize(x, y, shape=[shape[0], shape[1]])
    model.evaluate(x, y, batch_size=1)
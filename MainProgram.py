#from waitress import serve
from flask import Flask, render_template,request, send_from_directory
from scipy.misc import imsave, imread, imresize
import re
import sys 
import os
import cv2
import numpy as np
import tensorflow as tf
import time

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 196

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# initialize flask app
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def Main():
    return render_template("index.html")

@app.route('/analyze', methods=['GET', 'POST'])
def upload():
    # list to store result
    nameList = []
    # Define file path
    ImagePath = os.path.join(APP_ROOT, "images/")
    ResultPath = os.path.join(APP_ROOT, "result/")

    # Make the folder if it does not exists
    if not os.path.isdir(ImagePath):
        os.mkdir(ImagePath)
    
    if not os.path.isdir(ResultPath):
        os.mkdir(ResultPath)

    # Open the txt file to store result
    currTime = time.strftime("%Y%m%d-%H%M%S")
    Result = currTime + ".txt"
    ResultDestination = "/".join([ResultPath, Result])
    result = open(ResultDestination,'w')

    # Runs the detection for each image user has uploaded
    # Return the result on web page
    # Save the result in a txt file
    for file in request.files.getlist("file"):
        # Save the image user uploaded
        ImageName = file.filename
        ImageDestination = "/".join([ImagePath, ImageName])
        file.save(ImageDestination)
        print(ImageName)

        # Reads the image
        image = cv2.imread(ImageDestination)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Get the result
        name = category_index[classes[0][0]]
        Name = str(name)

        # Append the result into a tuple as list 
        # is not supported to be returned
        nameList.append(Name)
        nameTuple = tuple(nameList)
        response = str(nameTuple)

        # Write the result into txt file
        # It will only write the class'id for comparison purpose
        x = str(classes[0][0])
        result.write(x + "\n")
        print(Result)
    
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


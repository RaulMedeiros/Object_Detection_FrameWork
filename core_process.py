# -*- coding: utf-8 -*-
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import matplotlib
matplotlib.use('Agg')

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import argparse

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

def download_weights(MODEL_NAME, 
                     DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/',
                     directory = "models_Zoo"):

    MODEL_FILE = MODEL_NAME + '.tar.gz'

    if not os.path.exists(directory):
        os.makedirs(directory)

    ## Download Model
    file_name = DOWNLOAD_BASE + MODEL_FILE
    file_dst = directory+'/'+MODEL_FILE
    os.system("wget -N "+file_name+" -P "+directory)

    tar_file = tarfile.open('./'+directory+'/'+MODEL_FILE)

    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd()+'/'+directory)
    return True

def load_model(MODEL_NAME,directory = "models_Zoo"):
    ''' Load a (frozen) Tensorflow model into memory.'''

    PATH_TO_CKPT = directory+'/'+MODEL_NAME + '/frozen_inference_graph.pb'
    model = tf.Graph()

    with model.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return model
    
def load_label_map(PATH_TO_LABELS,NUM_CLASSES = 90):
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def rotate(src_img,rot_angle):
    # Size, in inches, of the output images.
    rows,cols,_ = src_img.shape
    # Build rotation matrix
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    # Perform Rotation
    return cv2.warpAffine(src_img,M,(cols,rows))

def process_frame(src_img,model,sess,category_index,display=True): 
    # Rotate if needed
    rot_angle = 0
    rot_img = rotate(src_img,rot_angle)

    # Resize image
    img_size = (800, 450) #(WIDTH,HEIGHT)
    image_np = cv2.resize(rot_img, img_size) 

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = model.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = model.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = model.get_tensor_by_name('detection_scores:0')
    classes = model.get_tensor_by_name('detection_classes:0')
    num_detections = model.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    if (display):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np
    else:

        n_detect = int(num_detections[0])
        boxes = boxes.tolist()[0][:n_detect]
        scores = scores.tolist()[0][:n_detect]
        class_detect = classes.tolist()[0][:n_detect]
        classes = [category_index[int(x)] for x in class_detect]
        return {"boxes" : boxes, 
                "scores" : scores,      
                "classes" : classes}

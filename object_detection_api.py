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

def segmentation(src_img,model,sess,category_index): 
  # Rotate if needed
  rot_angle = -40
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

import argparse


def main(args):

    # Donwload Model by its name
    download_weights(args['model']) 
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    model = load_model(args['model'])
    
    # List of the strings that is used to add correct label for each box.
    category_index = load_label_map(args['decoder'])

    cap = cv2.VideoCapture(args['source'])

    with model.as_default():
        with tf.Session(graph=model) as sess:
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Process image
                out_img = segmentation(frame,model,sess,category_index)

                #Plot image
                cv2.imshow('window',out_img)

                # Display the resulting frame
                cv2.imshow('window',out_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()  
    return True
  
if __name__ == '__main__':
    # Asterisk arguments
    parser = argparse.ArgumentParser(description='Object Detection')

    SOURCE_STREAM = "rtsp://admin:pvllck@10.110.1.56:554/cam/realmonitor?channel=1&subtype=1"
    parser.add_argument('--source', default=SOURCE_STREAM, type=str)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
    parser.add_argument('--decoder', default=PATH_TO_LABELS, type=str)

    # Name of the model to be downloaded and loaded 
    # MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
    # MODEL_NAME = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"
    # MODEL_NAME = "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18"
    #MODEL_NAME = "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
    MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
    # MODEL_NAME = "ssdlite_mobilenet_v2_coco_2018_05_09"
    #MODEL_NAME = "ssd_inception_v2_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_inception_v2_coco_2018_01_28"
    #MODEL_NAME = "faster_rcnn_resnet50_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_resnet50_lowproposals_coco_2018_01_28"
    #MODEL_NAME = "rfcn_resnet101_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_resnet101_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_resnet101_lowproposals_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28"
    #MODEL_NAME = "faster_rcnn_nas_coco_2018_01_28"
    # MODEL_NAME = "faster_rcnn_nas_lowproposals_coco_2018_01_28"
    # MODEL_NAME = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
    # MODEL_NAME = "mask_rcnn_inception_v2_coco_2018_01_28"
    # MODEL_NAME = "mask_rcnn_resnet101_atrous_coco_2018_01_28"
    # MODEL_NAME = "mask_rcnn_resnet50_atrous_coco_2018_01_28"

    parser.add_argument('--model', default=MODEL_NAME, type=str)

    args = vars(parser.parse_args())    
    main(args)

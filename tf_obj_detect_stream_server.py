#!/usr/bin/env python


# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
from flask import Flask, render_template, Response
import sys
import numpy

import tensorflow as tf

import core_process as cp

###########
app = Flask(__name__)
args = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
###########

def get_frame():
    global args

    # created a *threaded *video stream, allow the camera senor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")

    if( len(args['source']) == 1 ):
        src = int(args['source'])

    vs = WebcamVideoStream(src=src).start()

    # Donwload Model by its name
    cp.download_weights(args['model']) 
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    model = cp.load_model(args['model'])
    
    # List of the strings that is used to add correct label for each box.
    category_index = cp.load_label_map(args['decoder'])

    with model.as_default():
        with tf.Session(graph=model) as sess:
            while True:
                # grab the frame from the threaded video stream and process it
                src_frame = vs.read()
                    
                # Process image
                out_frame = cp.process_frame(src_frame,model,sess,category_index)

                imgencode=cv2.imencode('.jpg',out_frame)[1]
                stringData=imgencode.tostring()
                yield (b'--frame\r\n'
                    b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')    



    vs.stop()

if __name__ == '__main__':
    # Asterisk arguments
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--source', default="0", type=str)
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument("-p", "--port", action="store", default=8080, type=int)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
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
    
    app.run(host=args['host'], port=args['port'], debug=True, threaded=True)

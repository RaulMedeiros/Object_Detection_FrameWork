#!/usr/bin/env python

# import the necessary packages
from __future__ import print_function
from flask import Flask, render_template, Response, request, jsonify, url_for

import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from scipy import misc
import traceback

import os

import imutils
from imutils.video import WebcamVideoStream

import core_process as cp



###########
# Create and flask app

app = Flask(__name__)

# Render main html pages

@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/youtube')
def youtube():
    return render_template('youtube.html')

@app.route('/')
def index():
    return render_template('index.html')

##

# This subroutine activates youtube-dl with the input link, in the form inside youtube.html
# and, save the file into the /tmp folder, naming it  as "video_to_process.mp4"

@app.route('/video_url', methods=['GET', 'POST'])
def process_on_video():
    if request.method=='POST':
        print('aaaaaaaaaaaaaaaaauehauehuhaeuhaeuea')

        filename = get_youtube_video()
        def get_frame():
            with model.as_default():
                with tf.Session(graph=model) as sess:
                    cap = cv2.VideoCapture('./static/{}'.format(filename)) 
                    while cap.isOpened():
                        # grab the frame from the threaded video stream and process it
                        ret, src_frame = cap.read()
                        # Process image
                        out_frame = cp.process_frame(src_frame,model,sess,category_index, display=True)
                        imgencode=cv2.imencode('.jpg',out_frame)[1]
                        stringData=imgencode.tostring()
                        yield (b'--frame\r\n'
                            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')    

        return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
               
    # return render_template('youtube.html', video_name = filename)


def get_youtube_video():
    '''
    video_examples
    https://www.youtube.com/watch?v=PhFcl72nhiM (person talking)
    https://www.youtube.com/watch?v=jjlBnrzSGjc (traffic camera)
    https://www.youtube.com/watch?v=MiN_kgpkn-U (people walking)
    https://www.youtube.com/watch?v=GKuG4fftJdk (dogs playing 01)
    https://www.youtube.com/watch?v=GKuG4fftJdk (dogs playing 02)
    
    '''
    filename = 'video_to_process.mp4'

    if os.path.isfile(('/home/odj_detect_app/static/{}'.format(filename))):
        print('here UHAuhAUhUAHUHA')
        os.system('rm /home/odj_detect_app/static/{}'.format(filename))
    
    os.system('youtube-dl -f 18 -o ' + '"/home/odj_detect_app/static/video_to_process.%(ext)s" ' + request.form['video_url'])

    return filename



###########

@app.route('/calc')
def calc():
    def get_frame():
        vs = WebcamVideoStream(src=src).start()
        with model.as_default():
            with tf.Session(graph=model) as sess:
                while True:
                    # grab the frame from the threaded video stream and process it
                    src_frame = vs.read()
                        
                    # Process image
                    out_frame = cp.process_frame(src_frame,model,sess,category_index, display=True)

                    imgencode=cv2.imencode('.jpg',out_frame)[1]
                    stringData=imgencode.tostring()
                    yield (b'--frame\r\n'
                        b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')    
        vs.stop()

    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict_frame():
	if request.method=='POST':
		# get uploaded image file if it exists
		file = request.files['image']
		
		# read in file as raw pixels values
		src_frame = np.array(misc.imread(file))

        print("\n\nsrc_frame:",src_frame.shape,"\n\n")

        with model.as_default():
            with tf.Session(graph=model) as sess:   
                try:
                    #Process image
                    prediction = cp.process_frame(src_frame, model, sess, category_index, display=False)
                    return jsonify(prediction)
                except Exception, e:
                    return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def init():
    # created a *threaded *video stream, allow the camera senor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from webcam...")

    if( len(args['source']) == 1 ):
        src = int(args['source'])

    # Donwload Model by its name
    cp.download_weights(args['model']) 
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    model = cp.load_model(args['model'])
    
    # List of the strings that is used to add correct label for each box.
    category_index = cp.load_label_map(args['decoder'])

    return src,model,category_index

if __name__ == '__main__':
    # Asterisk arguments
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--source', default="0", type=str)
    parser.add_argument('--host', default='localhost', type=str)
    parser.add_argument("--port", action="store", default=8080, type=int)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'
    parser.add_argument('--decoder', default=PATH_TO_LABELS, type=str)

    # Name of the model to be downloaded and loaded 
    # MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"
    MODEL_NAME = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"
    # MODEL_NAME = "ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18"
    #MODEL_NAME = "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
    # MODEL_NAME = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
    #MODEL_NAME = "ssd_mobilenet_v2_coco_2018_03_29"
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
    
    #Init Model (Download and Load into memory)
    src,model,category_index = init()

    # Accessible at 
    app.run(host=args['host'], port=args['port'], debug=False, threaded=True)

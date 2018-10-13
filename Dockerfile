FROM "ubuntu"

RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /tensorflow/models
RUN apt-get install -y git python-pip
RUN pip install tensorflow
RUN apt-get install -y protobuf-compiler python-pil python-lxml
RUN pip install jupyter
RUN pip install matplotlib
RUN apt-get -y install wget

#Set TensorFlow Base Repository
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Opencv Installation
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install python-opencv -y
RUN cd /home/ && mkdir odj_detect_app && cd odj_detect_app 

ADD object_detection_api_.py /home/odj_detect_app/object_detection_api_.py

ENV PYTHONPATH=$PYTHONPATH:/home/odj_detect_app:/tensorflow/models/research/object_detection/:/tensorflow/models/research/object_detection/protos:/tensorflow/models/research/
WORKDIR /home/odj_detect_app




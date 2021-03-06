FROM "ubuntu"

RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /tensorflow/models
RUN apt-get install -y git python-pip

#TODO Tensorflow With GPU

RUN pip install tensorflow
RUN apt-get install -y protobuf-compiler python-pil python-lxml
RUN pip install jupyter
RUN pip install matplotlib
RUN apt-get -y install wget
RUN pip install imutils
RUN pip install flask
RUN pip install scipy

#Set TensorFlow Base Repository
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Opencv Installation #
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install python-opencv -y
RUN cd /home/ && mkdir odj_detect_app && cd odj_detect_app 

# Install youtube-dl

RUN apt-get install -y youtube-dl
RUN pip install --upgrade youtube-dl

# ADD python scripts

ADD object_detection_api.py /home/odj_detect_app/object_detection_api.py
ADD detect_object.py /home/odj_detect_app/detect_object.py

# ADD pre-downloaded models

ADD models_Zoo /home/odj_detect_app/models_Zoo

#s ADD image test folder

#ADD image_test /home/image_test

ADD templates /home/odj_detect_app/templates
ADD object_detection_api.py /home/odj_detect_app/object_detection_api.py
ADD core_process.py /home/odj_detect_app/core_process.py
ADD obj_detect_stream_server.py /home/odj_detect_app/obj_detect_stream_server.py

ENV PYTHONPATH=$PYTHONPATH:/home/odj_detect_app:/tensorflow/models/research/object_detection/:/tensorflow/models/research/object_detection/protos:/tensorflow/models/research/
WORKDIR /home/odj_detect_app
RUN mkdir static

# Run commands in container startup
CMD ["python", "obj_detect_stream_server.py"]



